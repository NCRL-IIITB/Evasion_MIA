"""
train_victim.py
===============
Train one victim of the v2 experiment.  Both victims use the unregularised
regime of v1 (no augmentation, no early stopping, last of 35 epochs):

  --training standard   C  std_noreg   — identical recipe to v1's C
  --training fgsm       D  fgsm_noreg  — adversarial training, recipe below

Shared by both (unchanged from v1's train_matched_victim.py):
  DenseNet-121 (ImageNet init) + Linear-ReLU-Dropout(0.3)-Linear head,
  AdamW (lr 1e-4, wd 1e-4), batch 64, 35 epochs, 3-epoch linear warm-up +
  cosine (eta_min 1e-6), grad-clip 5, pos_weight BCE, AMP, same train/val
  split, same seed → same head init and batch order, checkpoint every 5 epochs.

What changes for the FGSM victim (and why)
------------------------------------------
  * --method fgsm_rs (default): FGSM with a random start and a 1.25·ε step
    (Wong et al., ICLR 2020).  v1's plain FGSM showed catastrophic-overfitting
    symptoms (adversarial val AUC jumping 0.87 → 0.41 between epochs; evasion
    AUC non-monotonic in ε).  'nfgsm' and the v1 'fgsm' are available too.
  * --eps 4 (default): ε = 4/255 in PIXEL units, i.e. ≈ 0.07 in normalised
    units per channel.  v1 used 0.02 normalised ≈ 1.15/255, too weak to change
    the clean behaviour of the model.
  * --alpha 0.5 (default, as v1): loss = α·L(x) + (1-α)·L(x_adv).
    α = 0 gives pure (Madry-style) adversarial training.

Monitoring (does not change training): every epoch the clean val metrics, FGSM
val metrics on all val images, and PGD-5 val metrics on the first 1,024 val
images, all at --eps.  A large FGSM–PGD gap signals catastrophic overfitting.

Resume / deadline: same mechanism as v1 (train_state.pt after every epoch;
exit code 75 before an epoch that would not finish by --deadline).

Usage
-----
  python train_victim.py --training fgsm --method fgsm_rs --eps 4 --alpha 0.5 \
      --seed 42 --out_dir results_v2/victims/fgsm_noreg_seed42
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as v2  # noqa: E402

mc = v2.mc

# ─── Shared recipe (identical to v1) ──────────────────────────────────────────
NUM_EPOCHS    = 35
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 3
ETA_MIN       = 1e-6
GRAD_CLIP     = 5.0
CKPT_EVERY    = 5

MONITOR_PGD_N     = 1024
MONITOR_PGD_STEPS = 5
STATE_FILE        = "train_state.pt"
EXIT_PAUSED       = 75


class EpochSampler(torch.utils.data.Sampler):
    """Shuffle that depends only on (seed, epoch) — identical to v1."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist())

    def __len__(self):
        return self.n


def cpu_state(model):
    return {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, loader, device, criterion, adv=None):
    """(loss, element-wise accuracy, macro AUC) on clean or adversarial val images.
    adv = None | ('fgsm', eps255) | ('pgd', eps255, steps, generator)."""
    model.eval()
    total, correct, n = 0.0, 0, 0
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=v2.CHANNELS_LAST)
        labels = labels.to(device, non_blocking=True)
        if adv is not None and adv[0] == "fgsm":
            images = v2.fgsm(model, images, labels, adv[1], criterion, amp=True)
        elif adv is not None and adv[0] == "pgd":
            images = v2.pgd(model, images, labels, adv[1], adv[2], criterion, amp=True,
                            generator=adv[3])
        with torch.no_grad():
            with v2.amp_ctx(device):
                logits = model(images)
            logits = logits.float()
            total += criterion(logits, labels).item() * images.size(0)
            probs = torch.sigmoid(logits)
            correct += ((probs > 0.5).float() == labels).sum().item()
            n += images.size(0)
        ys.append(labels.cpu().numpy())
        ps.append(probs.cpu().numpy())
    return total / n, correct / (n * v2.NUM_CLASSES), mc.macro_auc(np.vstack(ys), np.vstack(ps))


# ─── One training epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, args, attack, gen):
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=v2.CHANNELS_LAST)
        labels = labels.to(device, non_blocking=True)
        bs = images.size(0)

        if args.training == "fgsm":
            # Step 1 — adversarial examples from the current model in train mode (as v1)
            adv_images = attack(model, images, labels, args.eps, criterion, True, gen)
            # Step 2 — α·clean + (1-α)·adversarial, gradients accumulated (as v1)
            optimizer.zero_grad(set_to_none=True)
            if args.alpha > 0:
                with v2.amp_ctx(device):
                    logits = model(images)
                    clean_loss = criterion(logits, labels)
                scaler.scale(args.alpha * clean_loss).backward()
            else:
                # pure adversarial training: clean logits only for the train metrics,
                # computed in eval mode so they do not touch the BatchNorm statistics
                model.eval()
                with torch.no_grad(), v2.amp_ctx(device):
                    logits = model(images)
                    clean_loss = criterion(logits, labels)
                model.train()
            with v2.amp_ctx(device):
                adv_loss = criterion(model(adv_images), labels)
            scaler.scale((1.0 - args.alpha) * adv_loss).backward()
            step_loss = args.alpha * clean_loss.detach() + (1.0 - args.alpha) * adv_loss.detach()
        else:
            optimizer.zero_grad(set_to_none=True)
            with v2.amp_ctx(device):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            step_loss = loss.detach()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():                        # train metrics from the clean logits (v1)
            probs = torch.sigmoid(logits.detach().float())
            correct += ((probs > 0.5).float() == labels).sum()
            loss_sum += step_loss.float() * bs
        n += bs
        ys.append(labels)
        ps.append(probs)
    auc = mc.macro_auc(torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy())
    return loss_sum.item() / n, correct.item() / (n * v2.NUM_CLASSES), auc


# ─── Output ───────────────────────────────────────────────────────────────────

def write_meta(path, args, extra):
    fgsm = args.training == "fgsm"
    meta = {
        # keys Victim_Model/api.py expects
        "architecture": "densenet121", "num_classes": v2.NUM_CLASSES, "img_size": mc.IMG_SIZE,
        "label_names": v2.DISEASE_CLASSES, "dropout": True,
        # design
        "experiment": "Matched_Experiments_v2",
        "cell": "fgsm_noreg" if fgsm else "std_noreg",
        "cell_id": args.cell_id,
        "training_type": "adversarial" if fgsm else "standard",
        "regime": "noreg", "augmentation": False, "seed": args.seed,
        "adversarial_method": args.method if fgsm else None,
        "train_eps255": args.eps if fgsm else None,
        "train_eps_normalised_per_channel": v2.eps_normalised(args.eps) if fgsm else None,
        "adversarial_alpha": args.alpha if fgsm else None,
        "monitor_eps255": args.eps,
        # shared recipe
        "epochs_max": args.epochs, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
        "scheduler": f"LinearWarmup({args.warmup}ep, start 0.1)+Cosine(eta_min={ETA_MIN})",
        "grad_clip": GRAD_CLIP, "pos_weight_used": True, "mixed_precision": True,
        "clamp": "valid image range (per channel)",
        "split_seed": mc.SPLIT_SEED, "torch_version": torch.__version__,
    }
    meta.update(extra)
    v2.write_json(path, meta)


def write_summary(path, title, meta, history):
    with open(path, "w") as f:
        f.write(f"{title}\n{'=' * 118}\n")
        for k in ("training_type", "adversarial_method", "train_eps255", "adversarial_alpha",
                  "seed", "epochs_trained", "selected_epoch", "final_train_acc", "final_val_acc",
                  "final_val_auc", "final_adv_val_auc", "final_pgd_val_auc", "memorization_gap"):
            f.write(f"{k:<20s}: {meta.get(k)}\n")
        f.write(f"\n{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'TrAUC':>7} {'VlLoss':>8} {'VlAcc':>7} "
                f"{'VlAUC':>7} {'FgsmAcc':>8} {'FgsmAUC':>8} {'PgdAcc':>7} {'PgdAUC':>7} "
                f"{'LR':>10} {'Gap%':>7}\n{'-' * 118}\n")
        for r in history:
            f.write(f"{r['epoch']:5d} {r['train_loss']:8.4f} {r['train_acc']:7.4f} "
                    f"{r['train_auc']:7.4f} {r['val_loss']:8.4f} {r['val_acc']:7.4f} "
                    f"{r['val_auc']:7.4f} {r['adv_val_acc']:8.4f} {r['adv_val_auc']:8.4f} "
                    f"{r['pgd_val_acc']:7.4f} {r['pgd_val_auc']:7.4f} {r['lr']:10.8f} "
                    f"{r['gap_pct']:+7.2f}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="v2 victim training")
    p.add_argument("--training", required=True, choices=["standard", "fgsm"])
    p.add_argument("--method", default=v2.DEFAULT_METHOD, choices=v2.METHODS)
    p.add_argument("--eps", type=float, default=v2.DEFAULT_EPS255, help="ε in /255 pixel units")
    p.add_argument("--alpha", type=float, default=v2.DEFAULT_ALPHA, help="weight of the clean loss")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=v2.DEFAULT_CACHE_DIR)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--deadline", type=float, default=None)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        sys.exit("ERROR: --alpha must be in [0, 1]")
    args.cell_id = v2.victim_cell_id("fgsm_noreg" if args.training == "fgsm" else "std_noreg",
                                     args.method, args.eps, args.alpha)

    args.epochs = 3 if args.smoke else NUM_EPOCHS
    args.warmup = 1 if args.smoke else WARMUP_EPOCHS
    ckpt_every = 1 if args.smoke else CKPT_EVERY
    pgd_n = 64 if args.smoke else MONITOR_PGD_N

    device = v2.get_device()
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    desc = (f"training={args.training} seed={args.seed}"
            + (f" method={args.method} eps={args.eps:g}/255 alpha={args.alpha:g}"
               if args.training == "fgsm" else ""))
    print(f"[INFO] {desc} → {args.out_dir}", flush=True)
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ── Data: identical split for every cell ──────────────────────────────────
    cache = mc.ImageCache(args.cache_dir)
    train_df, val_df, _ = mc.load_splits()
    if args.smoke:
        train_df, val_df = train_df.iloc[:512], val_df.iloc[:256]
    train_idx, val_idx = cache.index_of(train_df["path"]), cache.index_of(val_df["path"])
    train_y, val_y = mc.parse_labels(train_df), mc.parse_labels(val_df)
    print(f"[INFO] Train {len(train_df)} / val {len(val_df)} images; no augmentation, "
          f"no early stopping", flush=True)

    sampler = EpochSampler(len(train_idx), args.seed)
    tw = mc.loader_workers(args.num_workers)
    train_loader = DataLoader(mc.CachedDataset(cache, train_idx, train_y), batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=tw, pin_memory=v2.pin(device),
                              persistent_workers=tw > 0,
                              generator=torch.Generator().manual_seed(args.seed))
    vw = mc.loader_workers(4)
    val_loader = DataLoader(mc.CachedDataset(cache, val_idx, val_y), batch_size=128, shuffle=False,
                            num_workers=vw, pin_memory=v2.pin(device), persistent_workers=vw > 0)
    pgd_loader = DataLoader(mc.CachedDataset(cache, val_idx[:pgd_n], val_y[:pgd_n]), batch_size=64,
                            shuffle=False, num_workers=0)

    # ── Model / optimiser / schedule (identical to v1) ────────────────────────
    model = v2.build_victim().to(device).to(memory_format=v2.CHANNELS_LAST)
    criterion = nn.BCEWithLogitsLoss(pos_weight=v2.pos_weight_from(train_y).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                      total_iters=args.warmup),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=args.epochs - args.warmup, eta_min=ETA_MIN)],
        milestones=[args.warmup])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    attack = v2.make_train_attack(args.method) if args.training == "fgsm" else None
    gen = v2.make_generator(device, args.seed + 7)           # random starts of FGSM-RS / N-FGSM

    history = []
    hist_path = os.path.join(args.out_dir, "history.csv")
    state_path = os.path.join(args.out_dir, STATE_FILE)
    fields = ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc",
              "adv_val_loss", "adv_val_acc", "adv_val_auc", "pgd_val_loss", "pgd_val_acc",
              "pgd_val_auc", "lr", "gap_pct", "elapsed_s"]

    def write_history():
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in history:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    config = dict(training=args.training, method=args.method if args.training == "fgsm" else None,
                  eps=args.eps, alpha=args.alpha if args.training == "fgsm" else None,
                  seed=args.seed, epochs=args.epochs, smoke=args.smoke)
    start_epoch = 1
    if os.path.exists(state_path) and not args.no_resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st["config"] != config:
            sys.exit(f"ERROR: {state_path} belongs to a different run {st['config']}; "
                     "delete it or pass --no_resume")
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        history = st["history"]
        torch.set_rng_state(st["rng_torch"])
        if device.type == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"])
        if st.get("rng_gen") is not None:
            gen.set_state(st["rng_gen"])
        np.random.set_state(st["rng_numpy"])
        random.setstate(st["rng_python"])
        start_epoch = st["epoch"] + 1
        print(f"[RESUME] continuing after epoch {st['epoch']}", flush=True)
    write_history()

    def save_state(epoch):
        v2.torch_save(dict(config=config, epoch=epoch, model=cpu_state(model),
                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           scaler=scaler.state_dict(), history=history,
                           rng_torch=torch.get_rng_state(),
                           rng_cuda=torch.cuda.get_rng_state() if device.type == "cuda" else None,
                           rng_gen=gen.get_state(), rng_numpy=np.random.get_state(),
                           rng_python=random.getstate()), state_path)

    for epoch in range(start_epoch, args.epochs + 1):
        if args.deadline and history:
            need = 1.15 * max(r["elapsed_s"] for r in history[-3:]) + 120
            if time.time() + need > args.deadline:
                print(f"[PAUSED] {need / 60:.0f} min needed for epoch {epoch}; state saved after "
                      f"epoch {epoch - 1}. Rerun to continue.", flush=True)
                sys.exit(EXIT_PAUSED)
        sampler.set_epoch(epoch)
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = train_one_epoch(model, train_loader, optimizer, scaler,
                                                  criterion, device, args, attack, gen)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, device, criterion)
        av_loss, av_acc, av_auc = evaluate(model, val_loader, device, criterion,
                                           adv=("fgsm", args.eps))
        pg_loss, pg_acc, pg_auc = evaluate(model, pgd_loader, device, criterion,
                                           adv=("pgd", args.eps, MONITOR_PGD_STEPS,
                                                v2.make_generator(device, 1234)))
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_auc=tr_auc,
                   val_loss=vl_loss, val_acc=vl_acc, val_auc=vl_auc, adv_val_loss=av_loss,
                   adv_val_acc=av_acc, adv_val_auc=av_auc, pgd_val_loss=pg_loss,
                   pgd_val_acc=pg_acc, pgd_val_auc=pg_auc, lr=lr,
                   gap_pct=(tr_acc - vl_acc) * 100, elapsed_s=round(time.time() - t0, 1))
        history.append(row)
        write_history()
        print(f"  Epoch {epoch:02d}/{args.epochs} loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"vl_auc={vl_auc:.4f} fgsm_auc={av_auc:.4f} pgd_auc={pg_auc:.4f} "
              f"lr={lr:.2e} gap={row['gap_pct']:+.2f}% ({row['elapsed_s']:.0f}s)", flush=True)
        if epoch == start_epoch and device.type == "cuda":
            print(f"  [GPU] peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                  flush=True)
        if epoch % ckpt_every == 0:
            v2.torch_save(cpu_state(model),
                          os.path.join(args.out_dir, "checkpoints", f"epoch_{epoch:02d}.pth"))
        save_state(epoch)

    h = history[-1]
    v2.torch_save(cpu_state(model), os.path.join(args.out_dir, "model.pth"))
    extra = dict(selection_rule="last epoch (no early stopping)", selected_epoch=len(history),
                 early_stop_epoch=None, epochs_trained=len(history),
                 final_train_acc=h["train_acc"], final_val_acc=h["val_acc"],
                 final_val_auc=h["val_auc"], final_adv_val_auc=h["adv_val_auc"],
                 final_pgd_val_auc=h["pgd_val_auc"], final_pgd_val_acc=h["pgd_val_acc"],
                 memorization_gap=h["train_acc"] - h["val_acc"],
                 history_file="history.csv", n_train=len(train_df), n_val=len(val_df),
                 smoke=args.smoke)
    meta_path = os.path.join(args.out_dir, "model_meta.json")
    write_meta(meta_path, args, extra)
    with open(meta_path) as f:
        meta = json.load(f)
    write_summary(os.path.join(args.out_dir, "summary.txt"), f"Victim {desc}", meta, history)
    if os.path.exists(state_path):
        os.remove(state_path)
    print(f"[DONE] {desc} → {args.out_dir}/model.pth", flush=True)


if __name__ == "__main__":
    main()
