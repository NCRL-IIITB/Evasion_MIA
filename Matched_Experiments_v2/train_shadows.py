"""
train_shadows.py
================
Train one set of 8 shadow models for one victim and KEEP them (v1 deleted
them after each attack, so every new attack idea needed hours of retraining).

  --kind std : standard training — the shadows of the paper's Shadow-model MIA
  --kind at  : FGSM-RS adversarial training (ε = 2/255, α = 0.5) — the
               shadows of AFMIA.  The attacker's own generic choice: half the
               victim's ε, mirroring the paper (shadow 0.01 vs victim 0.02).

Unchanged from the paper / v1: 8 shadows cycling ResNet-18 / MobileNetV3-S /
EfficientNet-B0 / DenseNet-121 / ShuffleNetV2, 10,000 pool images each (the
same per-shadow samples as v1: RandomState(seed).choice in order), pseudo-
labels = victim probabilities thresholded at 0.5 (black-box queries), 25
epochs, batch 32, Adam lr 1e-3, pos_weight BCE, grad-clip 5, ImageNet init.
Changed: mixed precision (≈2× faster on T4; does not change the method) and
the AT shadows use FGSM-RS on the pixel scale instead of plain FGSM.

Output (<out_dir> = results_v2/attacks/<run>/shadows_<kind>):
  plan.json          architecture + training indices (into the pool) per shadow
  shadow_<i>.pth     weights, written as soon as a shadow is trained
  shadow_<i>.json    training log of that shadow
  DONE.json          all shadows trained

Resumable: finished shadows are skipped.  With --deadline the script exits
with code 75 before a shadow that would not finish in time.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as v2  # noqa: E402

mc = v2.mc

NUM_SHADOWS     = 8
SHADOW_SIZE     = 10_000
SHADOW_EPOCHS   = 25
SHADOW_BATCH    = 32
SHADOW_LR       = 1e-3
AT_METHOD       = "fgsm_rs"
AT_EPS255       = 2.0
AT_ALPHA        = 0.5
EXIT_PAUSED     = 75


def shadow_plan(n_pool, seed, n_shadows, size):
    """Same per-shadow pool samples as v1 (_train_shadow_models)."""
    rng = np.random.RandomState(seed)
    plan = []
    for i in range(n_shadows):
        idx = rng.choice(n_pool, size=min(size, n_pool), replace=False)
        plan.append({"i": i, "arch": v2.SHADOW_ARCHS[i % len(v2.SHADOW_ARCHS)],
                     "seed": seed + i, "train_indices": [int(j) for j in idx]})
    return plan


def train_one(arch, seed, cache, idx, labels, device, kind, epochs, log):
    model = v2.build_shadow(arch, seed).to(device)
    model.train()
    dl = v2.loader(cache, idx, labels, batch_size=SHADOW_BATCH, shuffle=True, workers=2,
                   device=device, generator=torch.Generator().manual_seed(seed))
    criterion = nn.BCEWithLogitsLoss(pos_weight=v2.pos_weight_from(labels).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=SHADOW_LR)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    attack = v2.make_train_attack(AT_METHOD) if kind == "at" else None
    gen = v2.make_generator(device, seed + 1000)
    losses = []
    for epoch in range(epochs):
        total, nb = 0.0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if kind == "at":
                x_adv = attack(model, x, y, AT_EPS255, criterion, True, gen)
                optimizer.zero_grad(set_to_none=True)
                with v2.amp_ctx(device):
                    loss = (AT_ALPHA * criterion(model(x), y)
                            + (1 - AT_ALPHA) * criterion(model(x_adv), y))
            else:
                optimizer.zero_grad(set_to_none=True)
                with v2.amp_ctx(device):
                    loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            nb += 1
        losses.append(total / max(nb, 1))
        log(f"      [{arch}] epoch {epoch + 1}/{epochs} loss {losses[-1]:.4f}")
    return model, losses


def main():
    p = argparse.ArgumentParser(description="Train one shadow set for one victim")
    p.add_argument("--kind", required=True, choices=["std", "at"])
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--attack_dir", required=True, help="results_v2/attacks/<run>")
    p.add_argument("--cache_dir", default=v2.DEFAULT_CACHE_DIR)
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--deadline", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    n_shadows, size, epochs = NUM_SHADOWS, SHADOW_SIZE, SHADOW_EPOCHS
    if args.smoke:
        n_shadows, size, epochs = 2, 200, 1
    out_dir = os.path.join(args.attack_dir, f"shadows_{args.kind}")
    os.makedirs(out_dir, exist_ok=True)
    device = v2.get_device()
    torch.backends.cudnn.benchmark = True
    log = lambda m: print(m, flush=True)  # noqa: E731

    cache = mc.ImageCache(args.cache_dir)
    pool, eval_m, eval_nm = v2.attack_split(args.seed, smoke=args.smoke)
    if args.splits_dir:
        mc.save_attack_split(os.path.join(args.splits_dir, f"attack_split_seed{args.seed}.json"),
                             args.seed, pool, eval_m, eval_nm)
    pool_idx = cache.index_of(pool)
    probs = v2.victim_pool_probs(args.attack_dir, args.victim_dir, cache, pool_idx, device)
    pseudo = (probs > 0.5).astype(np.float32)
    log(f"[SHADOWS:{args.kind}] pool={len(pool_idx)}  victim pseudo-label positives/image "
        f"{pseudo.sum(1).mean():.2f}")

    plan_path = os.path.join(out_dir, "plan.json")
    plan = shadow_plan(len(pool_idx), args.seed, n_shadows, size)
    old = v2.read_json(plan_path)
    if old is not None and old["shadows"] != plan:
        sys.exit(f"ERROR: {plan_path} differs from this run's plan — wrong output attached?")
    v2.write_json(plan_path, {"kind": args.kind, "seed": args.seed, "pool_size": len(pool_idx),
                              "epochs": epochs, "batch_size": SHADOW_BATCH, "lr": SHADOW_LR,
                              "at": ({"method": AT_METHOD, "eps255": AT_EPS255, "alpha": AT_ALPHA}
                                     if args.kind == "at" else None),
                              "shadows": plan})

    durations = {}
    for s in plan:
        ckpt = os.path.join(out_dir, f"shadow_{s['i']}.pth")
        info_path = os.path.join(out_dir, f"shadow_{s['i']}.json")
        if os.path.exists(ckpt) and os.path.exists(info_path):
            durations.setdefault(s["arch"], v2.read_json(info_path)["seconds"])
            log(f"  shadow {s['i'] + 1}/{len(plan)} ({s['arch']}) already trained")
            continue
        est = durations.get(s["arch"])
        if args.deadline and est and time.time() + 1.15 * est + 60 > args.deadline:
            log(f"[PAUSED] shadow {s['i'] + 1} ({s['arch']}) needs ~{est / 60:.0f} min — "
                "not enough time left; rerun to continue.")
            sys.exit(EXIT_PAUSED)
        t0 = time.time()
        idx = np.asarray(s["train_indices"])
        model, losses = train_one(s["arch"], s["seed"], cache, pool_idx[idx], pseudo[idx],
                                  device, args.kind, epochs, log)
        v2.torch_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
        secs = round(time.time() - t0, 1)
        durations[s["arch"]] = secs
        v2.write_json(info_path, {"i": s["i"], "arch": s["arch"], "kind": args.kind,
                                  "seconds": secs, "epoch_loss": losses})
        log(f"  shadow {s['i'] + 1}/{len(plan)} ({s['arch']}) trained in {secs / 60:.1f} min → saved")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    v2.write_json(os.path.join(out_dir, "DONE.json"),
                  {"kind": args.kind, "n_shadows": len(plan), "durations_s": durations})
    log(f"[SHADOWS:{args.kind}] all {len(plan)} shadows ready → {out_dir}")


if __name__ == "__main__":
    main()
