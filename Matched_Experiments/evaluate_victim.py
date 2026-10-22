"""
evaluate_victim.py
==================
Per-victim evaluations that need no shadow models.

1. FGSM evasion  — same protocol as fgsm_attack/attack_and_compare.py (paper
   Fig. 5): test set = all 33,412 non-member images; clean pass plus FGSM at
   ε ∈ {0.001, 0.002, 0.005, 0.01, 0.02} with unweighted BCE on the true
   labels, eval mode, fp32, clamp to [-3, 3]; flip rate = fraction of the 15
   thresholded labels that change.

2. Generalisation + threshold MIAs on the selected model AND on every saved
   training checkpoint (for the gap-matched analysis).  Members / non-members
   are the SAME 5k + 5k evaluation images the shadow attacks are scored on.
     loss        score = −BCE(f(x), y)                          Yeom et al. 2018
     confidence  score = mean_c |2·p_c − 1|                     label-free
     adv_loss    score = −BCE(f(x_adv), y), FGSM ε=0.02 on y    Song et al. 2019
     fluct_wb    score = −Σ_c |p_c(x) − p_c(x_adv)|, FGSM ε=0.01 on the
                 victim's own pseudo-labels: the white-box (oracle) version
                 of the AFMIA fluctuation signal
   AUC / TPR@low-FPR are threshold-free; "best_threshold_acc" picks the
   threshold on the evaluation set itself and is therefore optimistic.

Usage
-----
  python evaluate_victim.py --victim_dir results/victims/fgsm_noreg_seed42 \
      --seed 42 --out_dir results/eval/fgsm_noreg_seed42
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_common as mc

EVASION_EPSILONS = [0.001, 0.002, 0.005, 0.01, 0.02]   # attack_and_compare.py
EVASION_BATCH    = 64
PROBE_EPSILON    = 0.01   # AFMIA probing budget (run_fluctuation_attack.py default)
TRAIN_EPSILON    = 0.02   # victim adversarial-training budget
CHANNELS_LAST    = torch.channels_last


# ─── 1. FGSM evasion (paper protocol) ─────────────────────────────────────────

def _compute_metrics(preds, labels, probs):
    """Same as attack_and_compare.compute_metrics() (AUC over classes present)."""
    hamming = np.mean(preds != labels)
    exact_match = np.mean(np.all(preds == labels, axis=1))
    tp = np.sum((preds == 1) & (labels == 1), axis=0)
    fp = np.sum((preds == 1) & (labels == 0), axis=0)
    fn = np.sum((preds == 0) & (labels == 1), axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    auc = mc.macro_auc(labels, probs)   # == roc_auc_score(macro) when every class occurs
    return {"hamming_loss": float(hamming), "exact_match": float(exact_match),
            "f1_macro": float(np.mean(f1)), "auc_macro": float(auc)}


def _inference(model, loader, device, epsilon=None):
    criterion = nn.BCEWithLogitsLoss()
    preds, labels, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=CHANNELS_LAST)
        y = y.to(device, non_blocking=True)
        if epsilon is not None:
            x = mc.fgsm(model, x, y, epsilon, criterion, amp=False)
        with torch.no_grad():
            p = torch.sigmoid(model(x))
        preds.append((p > 0.5).float().cpu().numpy())
        labels.append(y.cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(probs)


def run_evasion(model, cache, device, smoke):
    _, _, nonmember_df = mc.load_splits()
    if smoke:
        nonmember_df = nonmember_df.sample(n=256, random_state=42).reset_index(drop=True)
    loader = DataLoader(
        mc.CachedDataset(cache, cache.index_of(nonmember_df["path"]), mc.parse_labels(nonmember_df)),
        batch_size=EVASION_BATCH, shuffle=False, num_workers=6, pin_memory=True,
    )
    clean_preds, clean_labels, clean_probs = _inference(model, loader, device)
    clean = _compute_metrics(clean_preds, clean_labels, clean_probs)
    _, _, f1_pc, _ = precision_recall_fscore_support(clean_labels, clean_preds,
                                                     average=None, zero_division=0)
    print(f"  [EVASION] clean AUC={clean['auc_macro']:.4f}", flush=True)
    adv = {}
    for eps in EVASION_EPSILONS:
        a_preds, a_labels, a_probs = _inference(model, loader, device, epsilon=eps)
        m = _compute_metrics(a_preds, a_labels, a_probs)
        flip = float(np.mean(clean_preds != a_preds))
        adv[str(eps)] = {"metrics": m, "flip_rate": flip,
                         "comparison": {k: m[k] - clean[k] for k in clean}}
        print(f"  [EVASION] eps={eps:<6} AUC={m['auc_macro']:.4f}  flip={flip:.4f}", flush=True)
    return {
        "test_set": f"{len(nonmember_df)} non-member images (patient-disjoint)",
        "protocol": "attack_and_compare.py: unweighted BCE on true labels, eval mode, fp32, clamp [-3,3]",
        "clean": clean,
        "clean_per_class_f1": dict(zip(mc.DISEASE_CLASSES, f1_pc.tolist())),
        "adversarial": adv,
    }


# ─── 2. Generalisation + threshold MIAs ───────────────────────────────────────

def _per_sample(model, cache, paths, labels, device):
    loader = DataLoader(mc.CachedDataset(cache, cache.index_of(paths), labels),
                        batch_size=128, shuffle=False, num_workers=6, pin_memory=True)
    bce_none, bce = nn.BCEWithLogitsLoss(reduction="none"), nn.BCEWithLogitsLoss()
    out = {k: [] for k in ("loss", "certainty", "adv_loss", "fluct", "probs", "labels")}
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=CHANNELS_LAST)
        y = y.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        logits = logits.float()
        p = torch.sigmoid(logits)
        x_probe = mc.fgsm(model, x, (p > 0.5).float(), PROBE_EPSILON, bce, amp=True)
        x_adv = mc.fgsm(model, x, y, TRAIN_EPSILON, bce, amp=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            p_probe = torch.sigmoid(model(x_probe).float())
            adv_logits = model(x_adv).float()
        out["loss"].append(bce_none(logits, y).mean(1).cpu().numpy())
        out["certainty"].append((2 * p - 1).abs().mean(1).cpu().numpy())
        out["fluct"].append((p - p_probe).abs().sum(1).cpu().numpy())
        out["adv_loss"].append(bce_none(adv_logits, y).mean(1).cpu().numpy())
        out["probs"].append(p.cpu().numpy())
        out["labels"].append(y.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _macro_auc(labels, probs):
    return mc.macro_auc(labels, probs)


def threshold_eval(model, cache, device, eval_m, eval_nm, y_m, y_nm):
    m = _per_sample(model, cache, eval_m, y_m, device)
    n = _per_sample(model, cache, eval_nm, y_nm, device)
    gen = {
        "member_loss": float(m["loss"].mean()), "nonmember_loss": float(n["loss"].mean()),
        "loss_gap": float(n["loss"].mean() - m["loss"].mean()),
        "member_auc": _macro_auc(m["labels"], m["probs"]),
        "nonmember_auc": _macro_auc(n["labels"], n["probs"]),
        "member_adv_loss": float(m["adv_loss"].mean()),
        "nonmember_adv_loss": float(n["adv_loss"].mean()),
        "adv_loss_gap": float(n["adv_loss"].mean() - m["adv_loss"].mean()),
        "member_fluct": float(m["fluct"].mean()), "nonmember_fluct": float(n["fluct"].mean()),
    }
    gen["auc_gap"] = gen["member_auc"] - gen["nonmember_auc"]
    attacks = {
        "loss":       mc.membership_scores_metrics(-m["loss"], -n["loss"]),
        "confidence": mc.membership_scores_metrics(m["certainty"], n["certainty"]),
        "adv_loss":   mc.membership_scores_metrics(-m["adv_loss"], -n["adv_loss"]),
        "fluct_wb":   mc.membership_scores_metrics(-m["fluct"], -n["fluct"]),
    }
    return {"generalization": gen, "threshold_attacks": attacks}


def _label_lookup(paths):
    """Labels for the attack-split paths, from the manifest."""
    train_df, _, nonmember_df = mc.load_splits()
    table = {os.path.basename(p): y for df in (train_df, nonmember_df)
             for p, y in zip(df["path"], mc.parse_labels(df))}
    return np.stack([table[os.path.basename(p)] for p in paths])


def main():
    p = argparse.ArgumentParser(description="Evasion + threshold-MIA evaluation of one victim")
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--seed", type=int, required=True, help="attack-split seed (= victim seed)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=mc.DEFAULT_CACHE_DIR)
    p.add_argument("--skip_checkpoints", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    cache = mc.ImageCache(args.cache_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.victim_dir, "model_meta.json")) as f:
        meta = json.load(f)
    t0 = time.time()

    model = mc.load_victim_weights(os.path.join(args.victim_dir, "model.pth"), device)
    model = model.to(memory_format=CHANNELS_LAST)

    # 1. Evasion on the selected model
    print(f"[EVAL] {args.victim_dir}: FGSM evasion …", flush=True)
    evasion = run_evasion(model, cache, device, args.smoke)
    evasion["victim_dir"] = args.victim_dir
    mc.write_json(os.path.join(args.out_dir, "evasion.json"), evasion)

    # 2. Threshold MIAs on the same eval split the shadow attacks use
    pool, eval_m, eval_nm = mc.attack_split_for_seed(args.seed, smoke=args.smoke)
    y_m, y_nm = _label_lookup(eval_m), _label_lookup(eval_nm)
    print(f"[EVAL] threshold MIAs on {len(eval_m)} members + {len(eval_nm)} non-members …",
          flush=True)
    result = {"victim_dir": args.victim_dir, "seed": args.seed,
              "selected_epoch": meta.get("selected_epoch"),
              "final": threshold_eval(model, cache, device, eval_m, eval_nm, y_m, y_nm),
              "checkpoints": []}
    fin = result["final"]
    print(f"  final: loss_gap={fin['generalization']['loss_gap']:+.4f}  "
          + "  ".join(f"{k}_AUC={v['auc']:.4f}" for k, v in fin["threshold_attacks"].items()),
          flush=True)

    ckpts = sorted(glob.glob(os.path.join(args.victim_dir, "checkpoints", "epoch_*.pth")))
    if not args.skip_checkpoints:
        for ck in ckpts:
            epoch = int(re.search(r"epoch_(\d+)", os.path.basename(ck)).group(1))
            m = mc.load_victim_weights(ck, device).to(memory_format=CHANNELS_LAST)
            r = threshold_eval(m, cache, device, eval_m, eval_nm, y_m, y_nm)
            r["epoch"] = epoch
            result["checkpoints"].append(r)
            print(f"  epoch {epoch:02d}: loss_gap={r['generalization']['loss_gap']:+.4f}  "
                  f"loss_AUC={r['threshold_attacks']['loss']['auc']:.4f}  "
                  f"fluct_wb_AUC={r['threshold_attacks']['fluct_wb']['auc']:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
    result["runtime_s"] = round(time.time() - t0, 1)
    mc.write_json(os.path.join(args.out_dir, "threshold_mia.json"), result)
    print(f"[EVAL] done in {result['runtime_s']:.0f}s → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
