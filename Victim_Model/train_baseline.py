"""
train_baseline.py
=================
Train a baseline DenseNet-121 victim model on the MEMBER-ONLY portion of the
NIH Chest X-ray dataset with modern training improvements.

Features
--------
  ✓ AdamW optimiser with linear warmup + cosine annealing LR schedule
  ✓ Mixed-precision training (AMP) on CUDA
  ✓ Data augmentation: RandomResizedCrop, HFlip, Rotation, ColorJitter
  ✓ Dropout(0.3) in classifier head
  ✓ Early stopping (patience=7) on validation macro AUC
  ✓ Gradient clipping (max_norm=5.0)
  ✓ Per-class pos_weight for class-imbalanced BCE loss
  ✓ Training curves saved as PNG (2×2 subplot)

Inputs
------
  Victim_Model/manifest.csv   (produced by prepare_dataset.py)

Outputs
-------
  Victim_Model/victim_baseline.pth
  Victim_Model/victim_baseline_meta.json
  Victim_Model/logs/victim_baseline_history.csv    ← per-epoch metrics
  Victim_Model/logs/victim_baseline_summary.txt    ← final summary
  Victim_Model/logs/victim_baseline_curves.png     ← training curves

Usage
-----
  python train_baseline.py                          # defaults (DenseNet-121, 30 epochs)
  python train_baseline.py --epochs 50 --arch resnet50
  python train_baseline.py --batch_size 32 --lr 5e-5 --arch efficientnet_b3
"""

import os
import sys
import ast
import csv
import json
import time
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score


# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.csv")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")

DISEASE_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema",   "Fibrosis",       "Effusion",     "Pneumonia",    "Pleural_Thickening",
    "Cardiomegaly","Nodule",          "Mass",         "Hernia",       "No Finding",
]
NUM_CLASSES = len(DISEASE_CLASSES)  # 15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Defaults (can be overridden via CLI)
NUM_EPOCHS    = 30
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
IMG_SIZE      = 224
VAL_FRACTION  = 0.15     # fraction of members held out for monitoring
RANDOM_SEED   = 42
WARMUP_EPOCHS = 3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7
GRAD_CLIP     = 5.0


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NIHDataset(Dataset):
    """PyTorch Dataset for NIH Chest X-rays with multi-label targets."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"  Warning: failed to load {row['path']}: {e}", flush=True)
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        # label_idx stored as string "[1,0,0,...]" in manifest.csv
        label_list = ast.literal_eval(row["label_idx"])
        return img, torch.tensor(label_list, dtype=torch.float32)


# ─── Model builders ───────────────────────────────────────────────────────────

def build_model(architecture: str, num_classes: int) -> nn.Module:
    """Build a pretrained model with a custom multi-label head (always uses Dropout)."""
    arch = architecture.lower()
    if arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(512, num_classes)
        )
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(512, num_classes)
        )
    elif arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            "Choose from: densenet121, resnet50, efficientnet_b3"
        )

    # Full fine-tune — all layers trainable
    for p in model.parameters():
        p.requires_grad = True

    return model


# ─── Eval ─────────────────────────────────────────────────────────────────────

def compute_pos_weight(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Compute per-class pos_weight = (num_neg / num_pos) for BCEWithLogitsLoss.

    This corrects for extreme class imbalance in the NIH dataset (e.g. 'No Finding'
    is positive in ~56% of images while rare diseases are <3%). Without this, the
    model collapses to predicting all-zero which looks like ~93% element-wise accuracy
    but is clinically useless and gives near-zero MIA signal.
    """
    label_matrix = np.array(
        [ast.literal_eval(row) for row in train_df["label_idx"]], dtype=np.float32
    )
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = len(label_matrix) - pos_counts
    # Clip to avoid division by zero; cap ratio at 10 to avoid extreme gradients
    pos_weight = np.clip(neg_counts / np.maximum(pos_counts, 1), 0.1, 10.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def evaluate(model, loader, device, num_classes, criterion) -> tuple[float, float, float]:
    """Compute average loss, element-wise accuracy, and mean AUC-ROC on a DataLoader."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_n       += images.size(0) * num_classes

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / (total_n / num_classes)
    accuracy = total_correct / total_n

    # Mean AUC-ROC across classes (skip classes with no positive samples)
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    try:
        auc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        auc = float("nan")

    return avg_loss, accuracy, auc


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str) -> None:
    """Generate a 2×2 subplot of training curves and save to PNG."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline Victim Model — Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train", marker=".")
    ax.plot(epochs, history["val_loss"],   label="Val",   marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], label="Train", marker=".")
    ax.plot(epochs, history["val_acc"],   label="Val",   marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1, 0]
    ax.plot(epochs, history["train_auc"], label="Train", marker=".")
    ax.plot(epochs, history["val_auc"],   label="Val",   marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Macro AUC-ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], label="LR", marker=".", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train baseline NIH victim model")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--arch", type=str, default="densenet121",
        help="Model architecture: densenet121 | resnet50 | efficientnet_b3"
    )
    args = parser.parse_args()

    # Output paths
    model_path   = os.path.join(BASE_DIR, "victim_baseline.pth")
    meta_path    = os.path.join(BASE_DIR, "victim_baseline_meta.json")
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_csv_path = os.path.join(LOGS_DIR, "victim_baseline_history.csv")
    log_txt_path = os.path.join(LOGS_DIR, "victim_baseline_summary.txt")
    curves_path  = os.path.join(LOGS_DIR, "victim_baseline_curves.png")

    # Reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Device:       {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU:          {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Mode:         baseline")
    print(f"[INFO] Architecture: {args.arch}")
    print(f"[INFO] Mixed prec.:  {use_amp}")
    print(f"[INFO] Output model: {model_path}")
    sys.stdout.flush()

    # ── Load manifest, keep only members ──────────────────────────────────────
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run prepare_dataset.py first."
        )
    manifest  = pd.read_csv(MANIFEST_PATH)
    member_df = manifest[manifest["split"] == "member"].reset_index(drop=True)

    # Infer num_classes from stored label vector
    sample_label = ast.literal_eval(member_df["label_idx"].iloc[0])
    num_classes  = len(sample_label)
    label_names  = DISEASE_CLASSES[:num_classes]

    print(f"[INFO] Member images: {len(member_df)}")
    print(f"[INFO] Num classes:   {num_classes}")
    print(f"[INFO] Class names:   {label_names}")
    sys.stdout.flush()

    # ── Train / val split (within members) ────────────────────────────────────
    n_val    = int(len(member_df) * VAL_FRACTION)
    shuffled = member_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df   = shuffled.iloc[:n_val].reset_index(drop=True)
    train_df = shuffled.iloc[n_val:].reset_index(drop=True)

    print(f"[INFO] Training on {len(train_df)} images, validating on {len(val_df)}.")
    sys.stdout.flush()

    pin = device.type == "cuda"

    # Augmented training transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    print("[INFO] Using augmented training transform (baseline mode).")

    train_ds     = NIHDataset(train_df, transform=train_transform)
    val_ds       = NIHDataset(val_df)                             # always plain
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=pin, persistent_workers=True,
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=pin, persistent_workers=True,
    )

    # ── pos_weight for class imbalance ─────────────────────────────────────────
    pos_weight = compute_pos_weight(train_df, num_classes).to(device)
    print(f"[INFO] pos_weight (first 5): {pos_weight[:5].cpu().numpy().round(2)}")

    # ── Model, optimizer, scheduler, criterion ────────────────────────────────
    model     = build_model(args.arch, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # AMP scaler (only used when CUDA is available)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"[INFO] Optimizer:     AdamW (lr={args.lr}, weight_decay={WEIGHT_DECAY})")
    print(f"[INFO] Scheduler:     LinearWarmup({WARMUP_EPOCHS}ep) + CosineAnnealing")
    print(f"[INFO] Regularisation: Dropout(0.3) + weight_decay={WEIGHT_DECAY} + augmentation")
    print(f"[INFO] Early stop:    patience={PATIENCE} on val macro AUC")
    print(f"[INFO] Gradient clip: max_norm={GRAD_CLIP}")
    sys.stdout.flush()

    # ── CSV log setup ─────────────────────────────────────────────────────────
    csv_file   = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "train_loss", "train_acc", "train_auc",
         "val_loss", "val_acc", "val_auc", "lr", "gap_pct", "elapsed_s"]
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[TRAIN] Starting training …", flush=True)
    history = {
        "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss": [], "val_acc": [], "val_auc": [], "lr": [],
    }

    best_val_auc      = -1.0
    best_epoch        = 0
    best_state_dict   = None
    best_train_acc    = 0.0
    best_val_acc      = 0.0
    patience_counter  = 0
    actual_epochs     = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        all_train_labels, all_train_probs = [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()

            epoch_loss    += loss.item() * images.size(0)
            probs          = torch.sigmoid(logits.detach())
            preds          = (probs > 0.5).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_n       += images.size(0) * num_classes

            all_train_labels.append(labels.cpu().numpy())
            all_train_probs.append(probs.cpu().numpy())

        # Compute train metrics
        train_loss = epoch_loss / (epoch_n / num_classes)
        train_acc  = epoch_correct / epoch_n

        y_true_train = np.vstack(all_train_labels)
        y_prob_train = np.vstack(all_train_probs)
        try:
            train_auc = roc_auc_score(y_true_train, y_prob_train, average="macro")
        except ValueError:
            train_auc = float("nan")

        # Validation
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device, num_classes, criterion)
        gap_pct    = (train_acc - val_acc) * 100
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        # Step scheduler
        scheduler.step()

        actual_epochs = epoch

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  train_auc={train_auc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_auc={val_auc:.4f}  lr={current_lr:.6f}  gap={gap_pct:+.2f}%  ({elapsed:.1f}s)",
            flush=True,
        )
        csv_writer.writerow(
            [epoch, round(train_loss, 6), round(train_acc, 6), round(train_auc, 6),
             round(val_loss, 6), round(val_acc, 6), round(val_auc, 6),
             round(current_lr, 8), round(gap_pct, 4), round(elapsed, 2)]
        )
        csv_file.flush()
        sys.stdout.flush()

        # Early stopping based on val macro AUC
        if val_auc > best_val_auc:
            best_val_auc    = val_auc
            best_val_acc    = val_acc
            best_train_acc  = train_acc
            best_epoch      = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"\n[EARLY STOP] No improvement in val AUC for {PATIENCE} epochs. "
                    f"Best val AUC = {best_val_auc:.4f} at epoch {best_epoch}.",
                    flush=True,
                )
                break

    csv_file.close()

    # ── Save best model + metadata ────────────────────────────────────────────
    if not args.no_early_stopping:
        if best_state_dict is not None:
            torch.save(best_state_dict, model_path)
        else:
            # Fallback: save current model if no best was recorded
            torch.save(model.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
        best_train_acc = history["train_acc"][-1] if "history" in locals() else best_train_acc
        best_val_acc = history["val_acc"][-1] if "history" in locals() else best_val_acc
        best_val_auc = history["val_auc"][-1] if "history" in locals() else best_val_auc
        best_epoch = actual_epochs

    memorization_gap = best_train_acc - best_val_acc

    meta = {
        "architecture":          args.arch,
        "mode":                  "baseline",
        "num_classes":           num_classes,
        "label_names":           label_names,
        "img_size":              IMG_SIZE,
        "imagenet_mean":         IMAGENET_MEAN,
        "imagenet_std":          IMAGENET_STD,
        "final_train_acc":       best_train_acc,
        "final_val_acc":         best_val_acc,
        "final_val_auc":         best_val_auc,
        "memorization_gap":      memorization_gap,
        "epochs":                args.epochs,
        "epochs_trained":        actual_epochs,
        "best_epoch":            best_epoch,
        "batch_size":            args.batch_size,
        "learning_rate":         args.lr,
        "optimizer":             "AdamW",
        "scheduler":             "LinearWarmup+CosineAnnealing",
        "warmup_epochs":         WARMUP_EPOCHS,
        "weight_decay":          WEIGHT_DECAY,
        "dropout":               True,
        "augmentation":          True,
        "augmentation_details":  "RandomResizedCrop+HFlip+Rotation+ColorJitter",
        "mixed_precision":       True,
        "pos_weight_used":       True,
        "grad_clip":             GRAD_CLIP,
        "early_stopping_patience": PATIENCE,
        "training_type":         "standard",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Training curves ───────────────────────────────────────────────────────
    plot_training_curves(history, curves_path)

    # ── Human-readable summary log ────────────────────────────────────────────
    with open(log_txt_path, "w") as f:
        f.write(f"Victim Model Training Summary (Baseline)\n")
        f.write(f"{'=' * 55}\n")
        f.write(f"Mode:             baseline\n")
        f.write(f"Architecture:     {args.arch}\n")
        f.write(f"Epochs (max):     {args.epochs}\n")
        f.write(f"Epochs trained:   {actual_epochs}\n")
        f.write(f"Best epoch:       {best_epoch}\n")
        f.write(f"Batch size:       {args.batch_size}\n")
        f.write(f"Learning rate:    {args.lr}\n")
        f.write(f"Optimizer:        AdamW\n")
        f.write(f"Scheduler:        LinearWarmup({WARMUP_EPOCHS}ep) + CosineAnnealing\n")
        f.write(f"Weight decay:     {WEIGHT_DECAY}\n")
        f.write(f"Dropout:          True (0.3)\n")
        f.write(f"Augmentation:     RandomResizedCrop+HFlip+Rotation+ColorJitter\n")
        f.write(f"Mixed precision:  {use_amp}\n")
        f.write(f"pos_weight:       True\n")
        f.write(f"Gradient clip:    {GRAD_CLIP}\n")
        f.write(f"Early stopping:   patience={PATIENCE} on val macro AUC\n\n")
        f.write(f"Best train_acc:   {best_train_acc:.4f}\n")
        f.write(f"Best val_acc:     {best_val_acc:.4f}\n")
        f.write(f"Best val_AUC:     {best_val_auc:.4f}\n")
        f.write(f"Memorization gap: {memorization_gap * 100:+.2f}%\n\n")
        f.write(f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'TrAUC':>7}  {'VlLoss':>8}  {'VlAcc':>7}  {'VlAUC':>7}  {'LR':>10}  {'Gap%':>6}\n")
        f.write(f"{'-' * 80}\n")
        for i, (tl, ta, tauc, vl, va, vc, lr) in enumerate(zip(
            history["train_loss"], history["train_acc"], history["train_auc"],
            history["val_loss"],   history["val_acc"],   history["val_auc"],
            history["lr"],
        ), start=1):
            gap = (ta - va) * 100
            f.write(f"  {i:3d}  {tl:8.4f}  {ta:7.4f}  {tauc:7.4f}  {vl:8.4f}  {va:7.4f}  {vc:7.4f}  {lr:10.8f}  {gap:+6.2f}\n")

    print("\n[DONE]")
    print(f"  Mode:             baseline")
    print(f"  Model:            {model_path}")
    print(f"  Metadata:         {meta_path}")
    print(f"  Training log:     {log_csv_path}")
    print(f"  Summary:          {log_txt_path}")
    print(f"  Curves:           {curves_path}")
    print(f"  Best train_acc:   {best_train_acc:.4f}")
    print(f"  Best val_acc:     {best_val_acc:.4f}")
    print(f"  Best val_AUC:     {best_val_auc:.4f}")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Memorization gap: {memorization_gap * 100:+.2f}%")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
