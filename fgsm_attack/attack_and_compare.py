"""
attack_and_compare.py
=====================
FGSM Evasion Attack evaluation on the NIH Chest X-ray victim models.

Evaluates how well a trained model withstands the Fast Gradient Sign Method
(FGSM) adversarial perturbation at multiple epsilon levels, and compares
clean vs. adversarial performance.

Data split
----------
Uses the SAME manifest.csv produced by prepare_dataset.py (70/30 patient-level
split).  The "non_member" split (images the victim never trained on) is used as
the test set for a fair evaluation of generalisation and adversarial robustness.

Outputs
-------
  fgsm_attack/logs/fgsm_<victim>_report.json     ← full metrics
  fgsm_attack/logs/fgsm_<victim>_results.txt      ← human-readable summary
  fgsm_attack/logs/fgsm_<victim>_comparison.md     ← markdown table

Usage
-----
  python fgsm_attack/attack_and_compare.py                          # both models
  python fgsm_attack/attack_and_compare.py --victim baseline        # baseline only
  python fgsm_attack/attack_and_compare.py --victim adversarial     # adversarial only
  python fgsm_attack/attack_and_compare.py --max-samples 2000       # limit test set
"""

import argparse
import ast
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# ─── Resolve paths ────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VICTIM_DIR   = os.path.join(PROJECT_ROOT, "Victim_Model")
MANIFEST_PATH = os.path.join(VICTIM_DIR, "manifest.csv")
LOGS_DIR     = os.path.join(SCRIPT_DIR, "logs")

# ─── Disease classes — MUST match the victim model training exactly ────────────

DISEASE_CLASSES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema',   'Fibrosis',      'Effusion',     'Pneumonia',    'Pleural_Thickening',
    'Cardiomegaly','Nodule',         'Mass',         'Hernia',       'No Finding',
]
NUM_CLASSES = len(DISEASE_CLASSES)  # 15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
BATCH_SIZE    = 64

# Victim model variants
VICTIM_VARIANTS = [
    {
        "key":        "baseline",
        "label":      "Baseline (best-practice standard)",
        "model_path": os.path.join(VICTIM_DIR, "victim_baseline.pth"),
        "meta_path":  os.path.join(VICTIM_DIR, "victim_baseline_meta.json"),
    },
    {
        "key":        "adversarial",
        "label":      "Adversarial (FGSM-trained defence)",
        "model_path": os.path.join(VICTIM_DIR, "victim_adversarial_eps002_noaug.pth"),
        "meta_path":  os.path.join(VICTIM_DIR, "victim_adversarial_meta.json"),
    },
]

# FGSM epsilon values to evaluate
EPSILONS = [0.001, 0.002, 0.005, 0.01, 0.02]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NIHDataset(Dataset):
    """PyTorch Dataset for NIH Chest X-rays with multi-label targets.

    Reads from a pandas DataFrame with columns:
        path      — absolute path to the image file
        label_idx — string repr of a list, e.g. "[0, 1, 0, …]"
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row["path"]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        label_idx = ast.literal_eval(row["label_idx"])
        label = torch.tensor(label_idx, dtype=torch.float32)
        return img, label


# ─── Model building ──────────────────────────────────────────────────────────

def build_model(architecture: str, num_classes: int, use_dropout: bool = True) -> nn.Module:
    """Build the exact same model architecture as the training scripts.

    Parameters
    ----------
    architecture : str
        One of 'densenet121', 'resnet50', 'efficientnet_b3'.
    num_classes : int
        Number of output classes (15 for NIH).
    use_dropout : bool
        Whether the head includes Dropout(0.3) — must match the trained model.
    """
    arch = architecture.lower()

    if arch == "densenet121":
        model = models.densenet121(weights=None)
        in_f = model.classifier.in_features
        if use_dropout:
            model.classifier = nn.Sequential(
                nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            model.classifier = nn.Linear(in_f, num_classes)

    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        in_f = model.fc.in_features
        if use_dropout:
            model.fc = nn.Sequential(
                nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            model.fc = nn.Linear(in_f, num_classes)

    elif arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_f = model.classifier[1].in_features
        if use_dropout:
            model.classifier = nn.Sequential(
                nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            model.classifier[1] = nn.Linear(in_f, num_classes)

    else:
        raise ValueError(f"Unsupported architecture: '{architecture}'")

    return model


def load_victim_model(variant: dict) -> tuple[nn.Module, dict]:
    """Load a victim model from its .pth and _meta.json files.

    Returns the model (in eval mode, on device) and the metadata dict.
    """
    model_path = variant["model_path"]
    meta_path  = variant["meta_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    architecture = meta.get("architecture", "densenet121")
    num_classes  = int(meta.get("num_classes", NUM_CLASSES))
    use_dropout  = bool(meta.get("dropout", False))

    model = build_model(architecture, num_classes, use_dropout)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, meta


# ─── Data loading from manifest ──────────────────────────────────────────────

def load_test_data_from_manifest(max_samples: int | None = None) -> pd.DataFrame:
    """Load non-member images from manifest.csv as the test set.

    This ensures the evasion attack is evaluated on the SAME split that was
    used during victim model training — the non-member partition (unseen data).

    Parameters
    ----------
    max_samples : int, optional
        Limit the number of test samples (for faster debugging runs).

    Returns
    -------
    df : pd.DataFrame with columns: path, label, label_idx, split
    """
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: {MANIFEST_PATH} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["split"] == "nonmember"].copy()

    if max_samples is not None and len(test_df) > max_samples:
        test_df = test_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  Test set: {len(test_df)} non-member images (from manifest.csv)")
    return test_df


# ─── FGSM Attack ──────────────────────────────────────────────────────────────

def fgsm_attack(model, images, labels, epsilon, criterion):
    """Generate FGSM adversarial examples.

    x_adv = x + epsilon * sign(∇_x J(θ, x, y))
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.data.sign()
    adv_images = images + epsilon * grad_sign

    # Clamp in normalised space (roughly [-3, 3] covers the full range
    # for ImageNet-normalised inputs)
    adv_images = torch.clamp(adv_images, -3, 3)

    return adv_images.detach()


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, loader, epsilon=None):
    """Run inference on a data loader, optionally applying FGSM perturbation.

    Returns (preds, labels, probs) as numpy arrays.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if epsilon is not None:
            images = fgsm_attack(model, images, labels, epsilon, criterion)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels, probs):
    """Compute overall evaluation metrics."""
    hamming = np.mean(preds != labels)
    exact_match = np.mean(np.all(preds == labels, axis=1))

    tp = np.sum((preds == 1) & (labels == 1), axis=0)
    fp = np.sum((preds == 1) & (labels == 0), axis=0)
    fn = np.sum((preds == 0) & (labels == 1), axis=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    try:
        auc = roc_auc_score(labels, probs, average='macro')
    except Exception:
        auc = 0.0

    return {
        "hamming_loss": float(hamming),
        "exact_match":  float(exact_match),
        "f1_macro":     float(np.mean(f1)),
        "auc_macro":    float(auc),
    }


def compute_per_class_metrics(preds, labels):
    """Compute per-class precision, recall, and F1 score."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    return {
        "precision": precision.tolist(),
        "recall":    recall.tolist(),
        "f1":        f1.tolist(),
        "classes":   DISEASE_CLASSES,
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_banner(text: str):
    print(flush=True)
    print("=" * 80, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 80, flush=True)


def run_evaluation_for_victim(variant: dict, test_df: pd.DataFrame) -> dict:
    """Run clean + adversarial evaluation for one victim model.

    Returns a dict with all results for JSON serialisation.
    """
    label = variant["label"]
    key   = variant["key"]

    print_banner(f"VICTIM: {label}")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, meta = load_victim_model(variant)
    except FileNotFoundError as e:
        print(f"  SKIPPING: {e}", flush=True)
        return {}

    print(f"  Architecture:    {meta.get('architecture', '?')}")
    print(f"  Dropout:         {meta.get('dropout', False)}")
    print(f"  Training type:   {meta.get('training_type', '?')}")
    print(f"  val_AUC:         {meta.get('final_val_auc', 0):.4f}")
    print(f"  Device:          {device}")
    sys.stdout.flush()

    # ── Build data loader ─────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = NIHDataset(test_df, transform=transform)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    # ── Clean evaluation ──────────────────────────────────────────────────────
    print(f"\n  [CLEAN] Running inference on {len(test_df)} test images …")
    clean_preds, clean_labels, clean_probs = run_inference(model, loader)
    clean_metrics = compute_metrics(clean_preds, clean_labels, clean_probs)
    clean_per_class = compute_per_class_metrics(clean_preds, clean_labels)

    print(f"  Clean results:")
    for k, v in clean_metrics.items():
        print(f"    {k}: {v:.4f}")

    # ── Adversarial evaluation at each epsilon ────────────────────────────────
    adversarial_results = {}
    for eps in EPSILONS:
        print(f"\n  [FGSM ε={eps}] Running adversarial inference …")
        adv_preds, adv_labels, adv_probs = run_inference(model, loader, epsilon=eps)
        adv_metrics = compute_metrics(adv_preds, adv_labels, adv_probs)
        flip_rate   = float(np.mean(clean_preds != adv_preds))

        print(f"  Results:")
        for k, v in adv_metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"    flip_rate: {flip_rate:.4f}")

        adversarial_results[str(eps)] = {
            "metrics":   adv_metrics,
            "flip_rate": flip_rate,
            "comparison": {
                k: adv_metrics[k] - clean_metrics[k]
                for k in clean_metrics
            },
        }

    result = {
        "victim_key":     key,
        "victim_label":   label,
        "victim_meta":    {k: v for k, v in meta.items()
                          if k not in ("imagenet_mean", "imagenet_std", "label_names")},
        "test_set_size":  len(test_df),
        "test_set_split": "nonmember (from manifest.csv, 70/30 patient-level)",
        "clean":          clean_metrics,
        "clean_per_class": clean_per_class,
        "adversarial":    adversarial_results,
        "epsilons":       EPSILONS,
    }

    return result


def save_results(all_results: list[dict], total_time: float):
    """Save all results to JSON, TXT summary, and markdown comparison table."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().isoformat()

    for result in all_results:
        if not result:
            continue

        key = result["victim_key"]

        # ── JSON report ───────────────────────────────────────────────────────
        json_path = os.path.join(LOGS_DIR, f"fgsm_{key}_report.json")
        report = {
            "timestamp": timestamp,
            "total_runtime_s": total_time,
            **result,
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  ✓ JSON report: {json_path}")

        # ── TXT summary ──────────────────────────────────────────────────────
        txt_path = os.path.join(LOGS_DIR, f"fgsm_{key}_results.txt")
        with open(txt_path, "w") as f:
            f.write(f"FGSM Evasion Attack Results — {result['victim_label']}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"Timestamp:     {timestamp}\n")
            f.write(f"Test set:      {result['test_set_size']} images ({result['test_set_split']})\n")
            f.write(f"Device:        {device}\n\n")

            f.write(f"CLEAN PERFORMANCE:\n")
            for k, v in result["clean"].items():
                f.write(f"  {k}: {v:.4f}\n")

            f.write(f"\nPER-CLASS F1 (clean):\n")
            for cls, f1 in zip(result["clean_per_class"]["classes"],
                               result["clean_per_class"]["f1"]):
                f.write(f"  {cls:22s}: {f1:.4f}\n")

            f.write(f"\nADVERSARIAL RESULTS:\n")
            for eps_str, adv in result["adversarial"].items():
                f.write(f"\n  ε = {eps_str}:\n")
                for k, v in adv["metrics"].items():
                    f.write(f"    {k}: {v:.4f}\n")
                f.write(f"    flip_rate: {adv['flip_rate']:.4f}\n")
                f.write(f"    Changes vs clean:\n")
                for k, v in adv["comparison"].items():
                    symbol = "↑" if v > 0 else "↓" if v < 0 else "="
                    f.write(f"      {k}: {v:+.4f} {symbol}\n")

            f.write(f"\nTotal runtime: {total_time:.1f}s\n")
        print(f"  ✓ TXT summary: {txt_path}")

        # ── Markdown comparison table ─────────────────────────────────────────
        md_path = os.path.join(LOGS_DIR, f"fgsm_{key}_comparison.md")
        sorted_eps = sorted(result["adversarial"].keys(), key=float)

        with open(md_path, "w") as f:
            f.write(f"## FGSM Evasion Attack: {result['victim_label']}\n\n")
            f.write(f"Test set: {result['test_set_size']} non-member images "
                    f"(manifest.csv, 70/30 patient-level split)\n\n")

            # Header
            header = "| Metric | Clean |"
            sep    = "| --- | ---: |"
            for eps in sorted_eps:
                header += f" ε={eps} |"
                sep    += " ---: |"
            f.write(header + "\n")
            f.write(sep + "\n")

            # Rows for each metric
            for metric_key in ["hamming_loss", "exact_match", "f1_macro", "auc_macro"]:
                row = f"| {metric_key} | {result['clean'][metric_key]:.4f} |"
                for eps in sorted_eps:
                    val = result["adversarial"][eps]["metrics"][metric_key]
                    row += f" {val:.4f} |"
                f.write(row + "\n")

            # Flip rate row
            row = "| flip_rate | — |"
            for eps in sorted_eps:
                val = result["adversarial"][eps]["flip_rate"]
                row += f" {val:.4f} |"
            f.write(row + "\n")

        print(f"  ✓ Markdown table: {md_path}")


def print_comparison_table(all_results: list[dict]):
    """Print a combined comparison table across all victims."""
    print_banner("FINAL COMPARISON TABLE")

    for result in all_results:
        if not result:
            continue

        print(f"\n  {result['victim_label']}")
        print(f"  Test set: {result['test_set_size']} non-member images\n")

        # Header
        eps_keys = sorted(result["adversarial"].keys(), key=float)
        header = f"  {'Metric':20s} {'Clean':>10s}"
        for eps in eps_keys:
            header += f" {'ε=' + eps:>10s}"
        print(header)
        print("  " + "-" * (20 + 10 + len(eps_keys) * 11))

        # Metric rows
        for metric_key in ["hamming_loss", "exact_match", "f1_macro", "auc_macro"]:
            row = f"  {metric_key:20s} {result['clean'][metric_key]:10.4f}"
            for eps in eps_keys:
                val = result["adversarial"][eps]["metrics"][metric_key]
                row += f" {val:10.4f}"
            print(row)

        # Flip rate
        row = f"  {'flip_rate':20s} {'—':>10s}"
        for eps in eps_keys:
            val = result["adversarial"][eps]["flip_rate"]
            row += f" {val:10.4f}"
        print(row)

    sys.stdout.flush()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FGSM Evasion Attack evaluation on NIH Chest X-ray victim models"
    )
    parser.add_argument(
        "--victim", type=str, default="both",
        choices=["both", "baseline", "adversarial"],
        help="Which victim model(s) to attack. Default: both",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit the number of test samples (for faster debugging runs)",
    )
    args = parser.parse_args()

    print_banner("FGSM Evasion Attack — NIH Chest X-ray")
    print(f"  Device: {device}")
    print(f"  Epsilons: {EPSILONS}")

    # 1. Load test data from manifest (non-member split)
    print("\n[SETUP] Loading test data from manifest.csv (non-member split) …")
    test_df = load_test_data_from_manifest(max_samples=args.max_samples)

    # 2. Select victim models
    if args.victim == "both":
        variants = VICTIM_VARIANTS
    else:
        variants = [v for v in VICTIM_VARIANTS if v["key"] == args.victim]

    # 3. Run evaluations
    grand_start = time.time()
    all_results = []

    for variant in variants:
        result = run_evaluation_for_victim(variant, test_df)
        all_results.append(result)

    total_time = time.time() - grand_start

    if not any(all_results):
        print("\nNo results — did you train the victim models first?", flush=True)
        return

    # 4. Print comparison table
    print_comparison_table(all_results)

    # 5. Save results
    print_banner("SAVING RESULTS")
    save_results(all_results, total_time)

    print(f"\n  Total runtime: {total_time:.1f}s")
    print_banner("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
