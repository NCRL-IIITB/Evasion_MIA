"""
train_matched_victim.py
=======================
ONE training script for every cell of the matched 2x2 control experiment.

                       │  --training standard        │  --training fgsm
  ─────────────────────┼─────────────────────────────┼──────────────────────────────
  --regime reg         │  A  std_reg                 │  B  fgsm_reg   (+ B2 robustES)
  (aug + early stop)   │                             │
  --regime noreg       │  C  std_noreg               │  D  fgsm_noreg
  (no aug, 35 epochs)  │                             │

Everything except the two switches is shared and identical:
  DenseNet-121 (ImageNet init) + Linear-ReLU-Dropout(0.3)-Linear head,
  AdamW (lr 1e-4, wd 1e-4), batch 64, 35 max epochs, 3-epoch linear warm-up +
  cosine annealing (eta_min 1e-6), grad-clip 5.0, class-balanced pos_weight
  BCE, AMP, same train/val split, same seed → same head init and batch order.

  --training fgsm : loss = 0.5·L(x) + 0.5·L(x_adv), x_adv = FGSM(ε=0.02) from
                    the current model (exactly train_adversarial.py's recipe)
  --regime reg    : RandomResizedCrop+HFlip+Rotation+ColorJitter, early
                    stopping (patience 7) on clean val macro-AUC, best epoch kept
  --regime noreg  : no augmentation, no early stopping, last epoch kept

For fgsm+reg the same run also yields B2: the checkpoint selected by
adversarial val AUC (Rice et al., ICML 2020).  Training continues until BOTH
patience counters have expired; the clean-ES checkpoint is frozen at the epoch
where clean early stopping would have stopped, so B is exactly what a
clean-ES-only run produces.

Differences from the original scripts (all applied to every cell equally):
  * the baseline previously used 30 max epochs and eta_min 0; now 35 / 1e-6
  * images come from the 224x224 cache, so training augmentation is applied
    to the 224 px image instead of the 1024 px original
  * validation runs under AMP; FGSM input gradients are computed with
    torch.autograd.grad (identical gradient, no parameter-gradient work)

Usage
-----
  python train_matched_victim.py --training fgsm --regime noreg --seed 42 \
      --out_dir results/victims/fgsm_noreg_seed42
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_common as mc


# ─── Shared recipe (identical for every cell) ─────────────────────────────────

NUM_EPOCHS    = 35
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 3
ETA_MIN       = 1e-6
PATIENCE      = 7
GRAD_CLIP     = 5.0
FGSM_EPSILON  = 0.02
FGSM_ALPHA    = 0.5
CKPT_EVERY    = 5

# Same augmentation as train_baseline.py (applied to uint8 tensors)
AUGMENTATION = transforms.Compose([
    transforms.RandomResizedCrop(mc.IMG_SIZE, scale=(0.85, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])

CHANNELS_LAST = torch.channels_last


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Same as the original scripts: clip(num_neg / num_pos, 0.1, 10)."""
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.1, 10.0), dtype=torch.float32)


def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}


# ─── Evaluation (clean or FGSM) ───────────────────────────────────────────────

def evaluate(model, loader, device, criterion, num_classes, adv_epsilon=None):
    """Return (loss, element-wise accuracy, macro AUC) on clean or FGSM inputs.

    FGSM uses the same pos_weight criterion and true labels, as in
    train_adversarial.evaluate_adversarial().
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=CHANNELS_LAST)
        labels = labels.to(device, non_blocking=True)
        if adv_epsilon is not None:
            images = mc.fgsm(model, images, labels, adv_epsilon, criterion, amp=True)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
            logits = logits.float()
            total_loss += criterion(logits, labels).item() * images.size(0)
            probs = torch.sigmoid(logits)
            correct += ((probs > 0.5).float() == labels).sum().item()
            n += images.size(0)
        ys.append(labels.cpu().numpy())
        ps.append(probs.cpu().numpy())
    return total_loss / n, correct / (n * num_classes), mc.macro_auc(np.vstack(ys), np.vstack(ps))


# ─── One training epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, training,
                    num_classes):
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    ys, ps = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=CHANNELS_LAST)
        labels = labels.to(device, non_blocking=True)
        bs = images.size(0)

        if training == "fgsm":
            # Step 1 — FGSM from the current model in train mode (as the original)
            x = images.detach().clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss_tmp = criterion(model(x), labels)
            (grad,) = torch.autograd.grad(scaler.scale(loss_tmp), x)
            adv_images = torch.clamp(
                images + FGSM_EPSILON * torch.nan_to_num(grad.sign(), nan=0.0), -3.0, 3.0
            ).detach()

            # Step 2 — alpha·clean + (1-alpha)·adversarial, gradients accumulated
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                clean_loss = criterion(logits, labels)
            scaler.scale(FGSM_ALPHA * clean_loss).backward()
            with torch.autocast("cuda", dtype=torch.float16):
                adv_loss = criterion(model(adv_images), labels)
            scaler.scale((1.0 - FGSM_ALPHA) * adv_loss).backward()
            step_loss = FGSM_ALPHA * clean_loss.detach() + (1.0 - FGSM_ALPHA) * adv_loss.detach()
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            step_loss = loss.detach()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Train metrics from the clean logits of this step (as the originals)
        with torch.no_grad():
            probs = torch.sigmoid(logits.detach().float())
            correct += ((probs > 0.5).float() == labels).sum()
            loss_sum += step_loss.float() * bs
        n += bs
        ys.append(labels)
        ps.append(probs)

    auc = mc.macro_auc(torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy())
    return loss_sum.item() / n, correct.item() / (n * num_classes), auc


# ─── Output helpers ───────────────────────────────────────────────────────────

def write_meta(path, args, extra: dict):
    meta = {
        # keys required by Victim_Model/api.py
        "architecture": "densenet121",
        "num_classes":  mc.NUM_CLASSES,
        "img_size":     mc.IMG_SIZE,
        "label_names":  mc.DISEASE_CLASSES,
        "dropout":      True,
        # design
        "training_type": "adversarial" if args.training == "fgsm" else "standard",
        "regime":        args.regime,
        "seed":          args.seed,
        "augmentation":  args.regime == "reg",
        "augmentation_details": ("RandomResizedCrop(0.85-1)+HFlip+Rotation(10)+ColorJitter"
                                 if args.regime == "reg" else None),
        # shared recipe
        "epochs_max":    args.epochs,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer":     "AdamW",
        "weight_decay":  WEIGHT_DECAY,
        "scheduler":     f"LinearWarmup({args.warmup}ep, start 0.1)+Cosine(eta_min={ETA_MIN})",
        "grad_clip":     GRAD_CLIP,
        "pos_weight_used": True,
        "mixed_precision": True,
        "adversarial_method":  "FGSM (Goodfellow et al. 2015)" if args.training == "fgsm" else None,
        "adversarial_epsilon": FGSM_EPSILON if args.training == "fgsm" else None,
        "adversarial_alpha":   FGSM_ALPHA if args.training == "fgsm" else None,
        "split_seed":    mc.SPLIT_SEED,
        "image_source":  "224x224 cache (Matched_Experiments/build_image_cache.py)",
        "torch_version": torch.__version__,
    }
    meta.update(extra)
    mc.write_json(path, meta)


def write_summary(path, title, meta, history):
    with open(path, "w") as f:
        f.write(f"{title}\n{'=' * 100}\n")
        for k in ("training_type", "regime", "seed", "augmentation", "selection_rule",
                  "epochs_max", "epochs_trained", "selected_epoch", "final_train_acc",
                  "final_val_acc", "final_val_auc", "final_adv_val_auc", "memorization_gap"):
            f.write(f"{k:<20s}: {meta.get(k)}\n")
        f.write(f"\n{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'TrAUC':>7} {'VlLoss':>8} "
                f"{'VlAcc':>7} {'VlAUC':>7} {'AdvLoss':>8} {'AdvAcc':>7} {'AdvAUC':>7} "
                f"{'LR':>10} {'Gap%':>7}\n{'-' * 100}\n")
        for r in history:
            f.write(f"{r['epoch']:5d} {r['train_loss']:8.4f} {r['train_acc']:7.4f} "
                    f"{r['train_auc']:7.4f} {r['val_loss']:8.4f} {r['val_acc']:7.4f} "
                    f"{r['val_auc']:7.4f} {r['adv_val_loss']:8.4f} {r['adv_val_acc']:7.4f} "
                    f"{r['adv_val_auc']:7.4f} {r['lr']:10.8f} {r['gap_pct']:+7.2f}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Matched 2x2 victim training")
    p.add_argument("--training", required=True, choices=["standard", "fgsm"])
    p.add_argument("--regime", required=True, choices=["reg", "noreg"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--robust_es_out_dir", default=None,
                   help="fgsm+reg only: where to write the adversarial-val-AUC-selected model (B2)")
    p.add_argument("--cache_dir", default=mc.DEFAULT_CACHE_DIR)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA is required.")
    if args.robust_es_out_dir and not (args.training == "fgsm" and args.regime == "reg"):
        sys.exit("ERROR: --robust_es_out_dir is only meaningful for --training fgsm --regime reg")

    args.epochs   = 3 if args.smoke else NUM_EPOCHS
    args.warmup   = 1 if args.smoke else WARMUP_EPOCHS
    patience      = 1 if args.smoke else PATIENCE
    ckpt_every    = 1 if args.smoke else CKPT_EVERY
    use_es        = args.regime == "reg"
    track_robust  = use_es and args.training == "fgsm" and args.robust_es_out_dir is not None

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Reproducibility: same seed → same head init and batch order in every cell
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    cell_desc = f"training={args.training} regime={args.regime} seed={args.seed}"
    print(f"[INFO] {cell_desc}  → {args.out_dir}", flush=True)
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ── Data (identical split for every cell) ─────────────────────────────────
    cache = mc.ImageCache(args.cache_dir)
    train_df, val_df, _ = mc.load_splits()
    if args.smoke:
        train_df, val_df = train_df.iloc[:512], val_df.iloc[:256]
    train_idx, val_idx = cache.index_of(train_df["path"]), cache.index_of(val_df["path"])
    train_y, val_y = mc.parse_labels(train_df), mc.parse_labels(val_df)
    print(f"[INFO] Train {len(train_df)} / val {len(val_df)} images "
          f"(split seed {mc.SPLIT_SEED})", flush=True)

    augment = AUGMENTATION if args.regime == "reg" else None
    print(f"[INFO] Augmentation: {'ON' if augment else 'OFF'}   "
          f"Early stopping: {'ON (patience %d, clean val AUC)' % patience if use_es else 'OFF'}"
          f"{'   + robust-ES tracking (adv val AUC)' if track_robust else ''}", flush=True)

    train_loader = DataLoader(
        mc.CachedDataset(cache, train_idx, train_y, uint8_transform=augment),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        mc.CachedDataset(cache, val_idx, val_y),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True, generator=torch.Generator().manual_seed(args.seed),
    )

    # ── Model / optimiser / schedule (identical for every cell) ───────────────
    model = mc.build_victim_model(mc.NUM_CLASSES).to(device).to(memory_format=CHANNELS_LAST)
    pos_weight = compute_pos_weight(train_y).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup, eta_min=ETA_MIN),
        ],
        milestones=[args.warmup],
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    hist_path = os.path.join(args.out_dir, "history.csv")
    fields = ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc",
              "val_auc", "adv_val_loss", "adv_val_acc", "adv_val_auc", "lr", "gap_pct",
              "elapsed_s"]
    hist_file = open(hist_path, "w", newline="")
    writer = csv.DictWriter(hist_file, fieldnames=fields)
    writer.writeheader()

    clean = dict(best=-1.0, epoch=0, wait=0, stopped=not use_es, state=None, stop_epoch=None)
    robust = dict(best=-1.0, epoch=0, wait=0, stopped=not track_robust, state=None, stop_epoch=None)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, args.training,
            mc.NUM_CLASSES)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, device, criterion, mc.NUM_CLASSES)
        av_loss, av_acc, av_auc = evaluate(model, val_loader, device, criterion, mc.NUM_CLASSES,
                                           adv_epsilon=FGSM_EPSILON)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_auc=tr_auc,
                   val_loss=vl_loss, val_acc=vl_acc, val_auc=vl_auc, adv_val_loss=av_loss,
                   adv_val_acc=av_acc, adv_val_auc=av_auc, lr=lr,
                   gap_pct=(tr_acc - vl_acc) * 100, elapsed_s=round(time.time() - t0, 1))
        history.append(row)
        writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        hist_file.flush()
        print(f"  Epoch {epoch:02d}/{args.epochs}  loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"tr_auc={tr_auc:.4f}  vl_acc={vl_acc:.4f} vl_auc={vl_auc:.4f} "
              f"adv_auc={av_auc:.4f}  lr={lr:.2e}  gap={row['gap_pct']:+.2f}%  "
              f"({row['elapsed_s']:.0f}s)", flush=True)
        if epoch == 1:
            print(f"  [GPU] peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                  flush=True)

        if epoch % ckpt_every == 0:
            torch.save(cpu_state(model), os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pth"))

        # Early stopping bookkeeping (clean val AUC → B; adversarial val AUC → B2)
        for tracker, value, name in ((clean, vl_auc, "clean"), (robust, av_auc, "robust")):
            if tracker["stopped"]:
                continue
            if value > tracker["best"]:
                tracker.update(best=value, epoch=epoch, wait=0, state=cpu_state(model))
            else:
                tracker["wait"] += 1
                if tracker["wait"] >= patience:
                    tracker.update(stopped=True, stop_epoch=epoch)
                    print(f"  [EARLY STOP:{name}] no improvement for {patience} epochs — "
                          f"best epoch {tracker['epoch']} (AUC={tracker['best']:.4f})",
                          flush=True)
        if use_es and clean["stopped"] and robust["stopped"]:
            break

    hist_file.close()
    epochs_trained = len(history)

    # ── Save the selected model(s) ────────────────────────────────────────────
    def finalize(out_dir, state, sel_epoch, rule, stop_epoch, title):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(state, os.path.join(out_dir, "model.pth"))
        h = history[sel_epoch - 1]
        extra = dict(
            selection_rule=rule, selected_epoch=sel_epoch,
            early_stop_epoch=stop_epoch, epochs_trained=epochs_trained,
            early_stopping_patience=patience if use_es else None,
            final_train_acc=h["train_acc"], final_val_acc=h["val_acc"],
            final_val_auc=h["val_auc"], final_adv_val_auc=h["adv_val_auc"],
            memorization_gap=h["train_acc"] - h["val_acc"],
            history_file=os.path.relpath(hist_path, out_dir),
            n_train=len(train_df), n_val=len(val_df), smoke=args.smoke,
        )
        write_meta(os.path.join(out_dir, "model_meta.json"), args, extra)
        with open(os.path.join(out_dir, "model_meta.json")) as f:
            meta = json.load(f)
        write_summary(os.path.join(out_dir, "summary.txt"), title, meta, history)
        print(f"[DONE] {title}: selected epoch {sel_epoch} ({rule}) → {out_dir}/model.pth",
              flush=True)

    for tracker in (clean, robust):          # never improved (all-NaN AUC) → last epoch
        if tracker["state"] is None:
            tracker.update(state=cpu_state(model), epoch=epochs_trained)

    if use_es:
        finalize(args.out_dir, clean["state"], clean["epoch"],
                 "best clean val macro-AUC (patience %d)" % patience,
                 clean["stop_epoch"], f"Victim {cell_desc} [clean-ES]")
        if track_robust:
            finalize(args.robust_es_out_dir, robust["state"], robust["epoch"],
                     "best adversarial val macro-AUC (FGSM eps=0.02, patience %d)" % patience,
                     robust["stop_epoch"], f"Victim {cell_desc} [robust-ES]")
    else:
        finalize(args.out_dir, cpu_state(model), epochs_trained,
                 "last epoch (no early stopping)", None, f"Victim {cell_desc}")


if __name__ == "__main__":
    main()
