"""
train_adversarial.py
====================
Adversarial training of a DenseNet-121 victim model on the MEMBER-ONLY portion
of the NIH Chest X-ray dataset, using FGSM adversarial training (Goodfellow
et al. 2015).

The training loop generates FGSM adversarial examples on-the-fly and trains on
a weighted combination of clean and adversarial losses:

    total_loss = alpha * clean_loss + (1 - alpha) * adversarial_loss

This makes the model robust to FGSM evasion attacks.  The hypothesis is that
adversarial training causes the model to memorize training-data patterns more
strongly, making it MORE susceptible to Membership Inference Attacks.

Inputs
------
  Victim_Model/manifest.csv   (produced by prepare_dataset.py)

Outputs
-------
  Victim_Model/victim_adversarial.pth
  Victim_Model/victim_adversarial_meta.json
  Victim_Model/logs/victim_adversarial_history.csv    ← per-epoch metrics
  Victim_Model/logs/victim_adversarial_summary.txt    ← final summary
  Victim_Model/logs/victim_adversarial_curves.png     ← training curves

Usage
-----
  python train_adversarial.py
  python train_adversarial.py --epsilon 0.03 --alpha 0.6 --epochs 40
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
NUM_EPOCHS    = 35      # adversarial training converges slower
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
IMG_SIZE      = 224
VAL_FRACTION  = 0.15     # fraction of members held out for monitoring
RANDOM_SEED   = 42


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

def _get_backbone(architecture: str) -> tuple["nn.Module", int]:
    """Return (backbone, in_features) for the given architecture string."""
    arch = architecture.lower()
    if arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        return model, model.classifier.in_features
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        return model, model.fc.in_features
    elif arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        return model, model.classifier[1].in_features
    else:
        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            "Choose from: densenet121, resnet50, efficientnet_b3"
        )


def build_model(architecture: str, num_classes: int) -> nn.Module:
    """Build a pretrained model with a custom multi-label head WITH Dropout(0.3).

    Adversarial training always uses the regularized head so that dropout=True
    in the metadata, which is required for api.py compatibility.
    """
    arch  = architecture.lower()
    model, in_f = _get_backbone(architecture)

    # Head WITH dropout — required for api.py compatibility
    head = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    if arch == "densenet121":
        model.classifier = head
    elif arch == "resnet50":
        model.fc = head
    elif arch == "efficientnet_b3":
        model.classifier = head

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


def evaluate_adversarial(
    model, loader, device, num_classes, criterion, epsilon: float
) -> tuple[float, float, float]:
    """Evaluate model on FGSM-perturbed validation images."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Generate adversarial examples
        x = images.clone().detach().requires_grad_(True)
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        grad_sign = x.grad.data.sign()
        adv_images = torch.clamp(images + epsilon * grad_sign, -3.0, 3.0).detach()

        # Evaluate on adversarial images
        with torch.no_grad():
            logits = model(adv_images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_n += images.size(0) * num_classes
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / (total_n / num_classes)
    accuracy = total_correct / total_n
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    try:
        auc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        auc = float("nan")
    return avg_loss, accuracy, auc


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str) -> None:
    """Save a 3×2 grid of training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Adversarial Training Curves", fontsize=15, fontweight="bold")

    # (0,0): Train loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-", linewidth=1.5)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1): Val loss
    axes[0, 1].plot(epochs, history["val_loss"], "r-", linewidth=1.5)
    axes[0, 1].set_title("Val Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # (1,0): Train/Val accuracy
    axes[1, 0].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=1.5)
    axes[1, 0].plot(epochs, history["val_acc"], "r-", label="Val", linewidth=1.5)
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1): Train/Val AUC + adversarial val AUC
    axes[1, 1].plot(epochs, history["train_auc"], "b-", label="Train AUC", linewidth=1.5)
    axes[1, 1].plot(epochs, history["val_auc"], "r-", label="Val AUC", linewidth=1.5)
    axes[1, 1].plot(
        epochs, history["adv_val_auc"], "r--", label="Adv Val AUC", linewidth=1.5
    )
    axes[1, 1].set_title("AUC-ROC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # (2,0): Learning rate
    axes[2, 0].plot(epochs, history["lr"], "g-", linewidth=1.5)
    axes[2, 0].set_title("Learning Rate")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("LR")
    axes[2, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[2, 0].grid(True, alpha=0.3)

    # (2,1): Memorization gap
    gap = [
        (ta - va) * 100
        for ta, va in zip(history["train_acc"], history["val_acc"])
    ]
    axes[2, 1].plot(epochs, gap, "m-", linewidth=1.5)
    axes[2, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2, 1].set_title("Memorization Gap (train_acc − val_acc)")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Gap (%)")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Training curves saved: {save_path}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial training (FGSM) for NIH victim model"
    )
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--arch", type=str, default="densenet121",
        help="Model architecture: densenet121 | resnet50 | efficientnet_b3"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.02,
        help="FGSM perturbation magnitude (default: 0.02)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for clean loss in combined objective; "
             "adversarial weight = 1 - alpha (default: 0.5)"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional suffix for output files, e.g. --tag overfit → "
             "victim_adversarial_overfit.pth (default: empty)"
    )
    parser.add_argument(
        "--no-augmentation", action="store_true",
        help="Disable data augmentation (use plain Resize + Normalize only)"
    )
    parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="Disable early stopping — train for all epochs and save best model"
    )
    args = parser.parse_args()

    epsilon = args.epsilon
    alpha   = args.alpha
    tag     = f"_{args.tag}" if args.tag else ""

    # ── Output paths ──────────────────────────────────────────────────────────
    model_path   = os.path.join(BASE_DIR, f"victim_adversarial{tag}.pth")
    meta_path    = os.path.join(BASE_DIR, f"victim_adversarial{tag}_meta.json")
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_csv_path  = os.path.join(LOGS_DIR, f"victim_adversarial{tag}_history.csv")
    log_txt_path  = os.path.join(LOGS_DIR, f"victim_adversarial{tag}_summary.txt")
    log_plot_path = os.path.join(LOGS_DIR, f"victim_adversarial{tag}_curves.png")

    # Reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Device:       {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU:          {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Architecture: {args.arch}")
    print(f"[INFO] Output model: {model_path}")
    print(f"[INFO] Training type: ADVERSARIAL (FGSM, Goodfellow et al. 2015)")
    print(f"[INFO] Epsilon:      {epsilon}")
    print(f"[INFO] Alpha:        {alpha}  (clean weight={alpha}, adv weight={1.0 - alpha})")
    print(f"[INFO] Mixed prec.:  {use_amp}")
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

    # Same augmentation as baseline for fair comparison (unless --no-augmentation)
    if args.no_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        print("[INFO] Augmentation DISABLED (--no-augmentation flag set).")
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        print("[INFO] Using augmented training transform.")

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

    # ── Model, optimizer, criterion ───────────────────────────────────────────
    model     = build_model(args.arch, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── LR schedule: LinearLR warmup (3 epochs) + CosineAnnealingLR ──────────
    warmup_epochs = 3
    warmup_scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[INFO] Regularisation: Dropout(0.3) + weight_decay=1e-4 + augmentation")
    print(f"[INFO] Optimizer:      AdamW (lr={args.lr}, wd=1e-4)")
    print(f"[INFO] Scheduler:      LinearWarmup({warmup_epochs}ep) + CosineAnnealing")
    print(f"[INFO] Early stopping: patience=7 on clean val AUC")
    sys.stdout.flush()

    # ── CSV log setup ─────────────────────────────────────────────────────────
    csv_file   = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_loss", "train_acc", "train_auc",
        "val_loss", "val_acc", "val_auc",
        "adv_val_loss", "adv_val_acc", "adv_val_auc",
        "lr", "gap_pct", "elapsed_s",
    ])

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[TRAIN] Starting adversarial training …", flush=True)
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss": [], "val_acc": [], "val_auc": [],
        "adv_val_loss": [], "adv_val_acc": [], "adv_val_auc": [],
        "lr": [],
    }

    best_val_auc    = -1.0
    best_epoch      = 0
    use_early_stop  = not args.no_early_stopping
    patience        = 7
    patience_ctr    = 0
    actual_epochs   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        all_train_labels, all_train_probs = [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Step 1: Generate FGSM adversarial examples ──
            # We need gradients w.r.t. input to compute perturbation direction
            x_for_grad = images.clone().detach().requires_grad_(True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits_temp = model(x_for_grad)
                    loss_temp = criterion(logits_temp, labels)
                # Scale loss for AMP before backward
                scaler.scale(loss_temp).backward()
                # Unscale the input gradients to get correct signs
                # Since we only need the SIGN, we can use the scaled gradients directly
                # (scaling doesn't change the sign)
                grad_sign = x_for_grad.grad.data.sign()
            else:
                logits_temp = model(x_for_grad)
                loss_temp = criterion(logits_temp, labels)
                loss_temp.backward()
                grad_sign = x_for_grad.grad.data.sign()

            # Generate adversarial images
            adv_images = torch.clamp(
                images + epsilon * grad_sign, -3.0, 3.0
            ).detach()

            # ── Step 2: Train on combined clean + adversarial loss ──
            # Use gradient accumulation to keep memory efficient:
            # only ONE set of activations in memory at a time
            optimizer.zero_grad()

            if use_amp:
                # Clean forward pass
                with torch.amp.autocast("cuda"):
                    clean_logits = model(images)
                    clean_loss = criterion(clean_logits, labels)
                scaler.scale(alpha * clean_loss).backward()

                # Adversarial forward pass (gradient accumulates)
                with torch.amp.autocast("cuda"):
                    adv_logits = model(adv_images)
                    adv_loss = criterion(adv_logits, labels)
                scaler.scale((1.0 - alpha) * adv_loss).backward()

                # Update
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                clean_logits = model(images)
                clean_loss = criterion(clean_logits, labels)
                (alpha * clean_loss).backward()

                adv_logits = model(adv_images)
                adv_loss = criterion(adv_logits, labels)
                ((1.0 - alpha) * adv_loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # Track metrics using clean_logits for consistency
            total_loss = alpha * clean_loss.item() + (1.0 - alpha) * adv_loss.item()
            epoch_loss += total_loss * images.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(clean_logits)
            preds = (probs > 0.5).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_n       += images.size(0) * num_classes
            all_train_labels.append(labels.cpu().numpy())
            all_train_probs.append(probs.detach().cpu().numpy())

        train_loss = epoch_loss / (epoch_n / num_classes)
        train_acc  = epoch_correct / epoch_n

        # Train AUC
        y_true_tr = np.vstack(all_train_labels)
        y_prob_tr = np.vstack(all_train_probs)
        try:
            train_auc = roc_auc_score(y_true_tr, y_prob_tr, average="macro")
        except ValueError:
            train_auc = float("nan")

        # Clean validation
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, device, num_classes, criterion
        )
        # Adversarial validation
        adv_val_loss, adv_val_acc, adv_val_auc = evaluate_adversarial(
            model, val_loader, device, num_classes, criterion, epsilon
        )

        # Step LR scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        gap_pct = (train_acc - val_acc) * 100
        elapsed = time.time() - t0
        actual_epochs = epoch

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["adv_val_loss"].append(adv_val_loss)
        history["adv_val_acc"].append(adv_val_acc)
        history["adv_val_auc"].append(adv_val_auc)
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:02d}/{args.epochs}  "
            f"loss={train_loss:.4f}  tr_acc={train_acc:.4f}  "
            f"vl_acc={val_acc:.4f}  vl_auc={val_auc:.4f}  "
            f"adv_auc={adv_val_auc:.4f}  lr={current_lr:.2e}  "
            f"gap={gap_pct:+.2f}%  ({elapsed:.1f}s)",
            flush=True,
        )
        csv_writer.writerow([
            epoch,
            round(train_loss, 6), round(train_acc, 6), round(train_auc, 6),
            round(val_loss, 6), round(val_acc, 6), round(val_auc, 6),
            round(adv_val_loss, 6), round(adv_val_acc, 6), round(adv_val_auc, 6),
            round(current_lr, 8), round(gap_pct, 4), round(elapsed, 2),
        ])
        csv_file.flush()
        sys.stdout.flush()

        # ── Early stopping on clean val AUC ───────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_ctr = 0
            # Save best model
            torch.save(model.state_dict(), model_path)
        else:
            patience_ctr += 1
            if use_early_stop and patience_ctr >= patience:
                print(
                    f"\n[EARLY STOP] No improvement in val AUC for {patience} "
                    f"epochs. Best epoch: {best_epoch} (AUC={best_val_auc:.4f})",
                    flush=True,
                )
                break

    csv_file.close()

    if not use_early_stop:
        torch.save(model.state_dict(), model_path)
        best_epoch = actual_epochs
    else:
        # ── Reload best model for final metrics ───────────────────────────────────
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_training_curves(history, log_plot_path)

    # ── Metadata ──────────────────────────────────────────────────────────────
    if not use_early_stop:
        final_train_acc   = history["train_acc"][-1]
        final_val_acc     = history["val_acc"][-1]
        final_val_auc     = history["val_auc"][-1]
        final_adv_val_auc = history["adv_val_auc"][-1]
    else:
        final_train_acc   = history["train_acc"][best_epoch - 1]
        final_val_acc     = history["val_acc"][best_epoch - 1]
        final_val_auc     = history["val_auc"][best_epoch - 1]
        final_adv_val_auc = history["adv_val_auc"][best_epoch - 1]
        
    memorization_gap = final_train_acc - final_val_acc

    meta = {
        "architecture":     args.arch,
        "mode":             "adversarial",
        "num_classes":      num_classes,
        "label_names":      label_names,
        "img_size":         IMG_SIZE,
        "imagenet_mean":    IMAGENET_MEAN,
        "imagenet_std":     IMAGENET_STD,
        "final_train_acc":  final_train_acc,
        "final_val_acc":    final_val_acc,
        "final_val_auc":    final_val_auc,
        "final_adv_val_auc": final_adv_val_auc,
        "memorization_gap": memorization_gap,
        "epochs":           args.epochs,
        "epochs_trained":   actual_epochs,
        "best_epoch":       best_epoch,
        "batch_size":       args.batch_size,
        "learning_rate":    args.lr,
        "optimizer":        "AdamW",
        "scheduler":        "LinearWarmup+CosineAnnealing",
        "warmup_epochs":    warmup_epochs,
        "weight_decay":     1e-4,
        "dropout":          True,   # CRITICAL: must be True for api.py compatibility
        "augmentation":     not args.no_augmentation,
        "augmentation_details": "None" if args.no_augmentation else "RandomResizedCrop+HFlip+Rotation+ColorJitter",
        "mixed_precision":  True,
        "pos_weight_used":  True,
        "grad_clip":        5.0,
        "early_stopping_patience": patience if use_early_stop else "disabled",
        "training_type":    "adversarial",
        "adversarial_epsilon": args.epsilon,
        "adversarial_alpha":   args.alpha,
        "adversarial_method":  "FGSM (Goodfellow et al. 2015)",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Human-readable summary log ────────────────────────────────────────────
    with open(log_txt_path, "w") as f:
        f.write(f"Adversarial Victim Model Training Summary\n")
        f.write(f"{'=' * 55}\n")
        f.write(f"Training type:    adversarial (FGSM)\n")
        f.write(f"Architecture:     {args.arch}\n")
        f.write(f"Epochs:           {args.epochs} (trained {actual_epochs})\n")
        f.write(f"Best epoch:       {best_epoch}\n")
        f.write(f"Batch size:       {args.batch_size}\n")
        f.write(f"Learning rate:    {args.lr}\n")
        f.write(f"Weight decay:     1e-4\n")
        f.write(f"Dropout:          True\n")
        f.write(f"pos_weight:       True\n")
        f.write(f"Gradient clip:    5.0\n")
        f.write(f"FGSM epsilon:     {epsilon}\n")
        f.write(f"Alpha (clean wt): {alpha}\n\n")
        f.write(f"Final train_acc:  {final_train_acc:.4f}\n")
        f.write(f"Final val_acc:    {final_val_acc:.4f}\n")
        f.write(f"Final val_AUC:    {final_val_auc:.4f}\n")
        f.write(f"Final adv_val_AUC:{final_adv_val_auc:.4f}\n")
        f.write(f"Memorization gap: {memorization_gap * 100:+.2f}%\n\n")
        header = (
            f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'TrAUC':>7}  "
            f"{'VlLoss':>8}  {'VlAcc':>7}  {'VlAUC':>7}  "
            f"{'AdvLoss':>8}  {'AdvAcc':>7}  {'AdvAUC':>7}  {'Gap%':>6}\n"
        )
        f.write(header)
        f.write(f"{'-' * 100}\n")
        for i, (tl, ta, tauc, vl, va, vc, avl, ava, avc) in enumerate(zip(
            history["train_loss"], history["train_acc"], history["train_auc"],
            history["val_loss"],   history["val_acc"],   history["val_auc"],
            history["adv_val_loss"], history["adv_val_acc"], history["adv_val_auc"],
        ), start=1):
            gap = (ta - va) * 100
            f.write(
                f"  {i:3d}  {tl:8.4f}  {ta:7.4f}  {tauc:7.4f}  "
                f"{vl:8.4f}  {va:7.4f}  {vc:7.4f}  "
                f"{avl:8.4f}  {ava:7.4f}  {avc:7.4f}  {gap:+6.2f}\n"
            )

    print("\n[DONE]")
    print(f"  Training type:    adversarial (FGSM, Goodfellow et al. 2015)")
    print(f"  Model:            {model_path}")
    print(f"  Metadata:         {meta_path}")
    print(f"  Training log:     {log_csv_path}")
    print(f"  Summary:          {log_txt_path}")
    print(f"  Curves:           {log_plot_path}")
    print(f"  Final train_acc:  {final_train_acc:.4f}")
    print(f"  Final val_acc:    {final_val_acc:.4f}")
    print(f"  Final val_AUC:    {final_val_auc:.4f}")
    print(f"  Final adv_val_AUC:{final_adv_val_auc:.4f}")
    print(f"  Memorization gap: {memorization_gap * 100:+.2f}%")
    print(f"  Best epoch:       {best_epoch}")
    print(f"\n  Clean vs Adversarial val AUC comparison:")
    print(f"    Clean val AUC:       {final_val_auc:.4f}")
    print(f"    Adversarial val AUC: {final_adv_val_auc:.4f}")
    delta = final_val_auc - final_adv_val_auc
    print(f"    Delta (clean - adv): {delta:+.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
