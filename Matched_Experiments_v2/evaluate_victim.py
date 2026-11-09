"""
evaluate_victim.py
==================
Per-victim evaluations that need no shadow models.

1. Evasion (the third attack of the comparison)
   * FGSM on all 33,412 non-member images at ε ∈ {0.5, 1, 2, 4, 8}/255
     (pixel units), unweighted BCE on the true labels, eval mode, clamped to
     the valid image range.  Flip rate = fraction of the 15 thresholded labels
     that change (same definition as the paper / v1).
   * PGD-10 (step ε/4, random start) at ε ∈ {1, 2, 4}/255 on a fixed random
     subset of 5,000 non-members — FGSM-only robustness can be gradient
     masking, PGD shows whether it is real.

2. White-box reference MIAs (threshold attacks, no shadow models) on the same
   5k + 5k evaluation images the shadow attacks use — for the final model and
   every saved checkpoint (gap-matched analysis):
     loss         −BCE(f(x), y)                                   Yeom et al. 2018
     confidence   mean_c |2·p_c − 1|                              label-free
     adv_loss     −BCE(f(x_adv), y), FGSM at --probe_eps on y     Song et al. 2019
     adv_loss_pgd −BCE(f(x_adv), y), PGD-5 at --probe_eps         (final model only)
     fluct_e{1,4} −Σ_c |p_c(x) − p_c(x_adv)|, FGSM on the model's own predictions
     dlogit_e4    −mean_c |z_c(x) − z_c(x_adv)| (logit space)
   Per-sample scores of the final model are saved (threshold_scores.npz) so
   C and D can be compared with a paired bootstrap after merging.

Resumable: evasion.json and every finished checkpoint are kept; a rerun
continues from there.

Usage
-----
  python evaluate_victim.py --victim_dir results_v2/victims/fgsm_noreg_seed42 \
      --seed 42 --out_dir results_v2/eval/fgsm_noreg_seed42
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as v2  # noqa: E402

mc = v2.mc

PAPER_EPS_NORM   = [0.001, 0.002, 0.005, 0.01, 0.02]   # paper Table I / Fig. 6 (normalised units)
PAPER_FLUCT_EPS  = 0.01                                 # paper Eq. 7 (normalised units)
EVASION_FGSM_EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
EVASION_PGD_EPS  = [1.0, 2.0, 4.0]
PGD_STEPS        = 10
PGD_SUBSET       = 5000
FLUCT_EPS        = [1.0, 4.0]
DEFAULT_PROBE    = 4.0
THRESH_PGD_STEPS = 5


# ─── 1. Evasion ───────────────────────────────────────────────────────────────

def _metrics(preds, labels, probs):
    tp = np.sum((preds == 1) & (labels == 1), axis=0)
    fp = np.sum((preds == 1) & (labels == 0), axis=0)
    fn = np.sum((preds == 0) & (labels == 1), axis=0)
    prec, rec = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"hamming_loss": float(np.mean(preds != labels)),
            "exact_match": float(np.mean(np.all(preds == labels, axis=1))),
            "f1_macro": float(np.mean(f1)), "auc_macro": float(mc.macro_auc(labels, probs))}


def _inference(model, loader, device, attack=None):
    crit = nn.BCEWithLogitsLoss()
    preds, labels, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=v2.CHANNELS_LAST)
        y = y.to(device, non_blocking=True)
        if attack is not None:
            x = attack(x, y, crit)
        with torch.no_grad(), v2.amp_ctx(device):
            p = torch.sigmoid(model(x).float())
        preds.append((p > 0.5).float().cpu().numpy())
        labels.append(y.cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(probs)


def run_evasion(model, cache, device, smoke):
    _, _, nm = mc.load_splits()
    if smoke:
        nm = nm.sample(n=min(96, len(nm)), random_state=42).reset_index(drop=True)
    idx, y = cache.index_of(nm["path"]), mc.parse_labels(nm)
    dl = v2.loader(cache, idx, y, batch_size=64, workers=6, device=device)
    c_pred, c_lab, c_prob = _inference(model, dl, device)
    clean = _metrics(c_pred, c_lab, c_prob)
    print(f"  [EVASION] clean AUC={clean['auc_macro']:.4f} on {len(idx)} non-members", flush=True)
    out = {"test_set": f"{len(idx)} non-member images (patient-disjoint)",
           "protocol": "unweighted BCE on true labels, eval mode, ε in /255 pixel units, "
                       "clamped to the valid image range",
           "paper_protocol": "paper Table I: ε in normalised units, clamp to [-3, 3]",
           "clean": clean, "paper_fgsm": {}, "fgsm": {}, "pgd": {}}
    for e in PAPER_EPS_NORM:                         # the paper's own grid (Table I, Fig. 6)
        a_pred, a_lab, a_prob = _inference(
            model, dl, device,
            attack=lambda x, t, c, e=e: v2.fgsm_v1(x, v2.input_grad(model, x, t, c).sign(), e))
        m = _metrics(a_pred, a_lab, a_prob)
        flip = float(np.mean(c_pred != a_pred))
        out["paper_fgsm"][f"{e:g}"] = {"metrics": m, "flip_rate": flip}
        print(f"  [EVASION] paper FGSM ε={e:g} (normalised)  AUC={m['auc_macro']:.4f}  "
              f"flip={flip:.4f}", flush=True)
    for e in EVASION_FGSM_EPS:
        a_pred, a_lab, a_prob = _inference(
            model, dl, device, attack=lambda x, t, c, e=e: v2.fgsm(model, x, t, e, c))
        m = _metrics(a_pred, a_lab, a_prob)
        flip = float(np.mean(c_pred != a_pred))
        out["fgsm"][f"{e:g}"] = {"metrics": m, "flip_rate": flip}
        print(f"  [EVASION] FGSM ε={e:g}/255  AUC={m['auc_macro']:.4f}  flip={flip:.4f}",
              flush=True)

    rng = np.random.RandomState(0)
    sub = np.sort(rng.choice(len(idx), size=min(24 if smoke else PGD_SUBSET, len(idx)),
                             replace=False))
    dls = v2.loader(cache, idx[sub], y[sub], batch_size=64, workers=6, device=device)
    s_pred, s_lab, s_prob = c_pred[sub], c_lab[sub], c_prob[sub]
    out["pgd_subset"] = {"n": int(len(sub)), "clean": _metrics(s_pred, s_lab, s_prob)}
    for e in EVASION_PGD_EPS:
        gen = v2.make_generator(device, 99)
        a_pred, a_lab, a_prob = _inference(
            model, dls, device,
            attack=lambda x, t, c, e=e, gen=gen: v2.pgd(model, x, t, e, PGD_STEPS, c, generator=gen))
        m = _metrics(a_pred, a_lab, a_prob)
        flip = float(np.mean(s_pred != a_pred))
        out["pgd"][f"{e:g}"] = {"metrics": m, "flip_rate": flip}
        print(f"  [EVASION] PGD-{PGD_STEPS} ε={e:g}/255  AUC={m['auc_macro']:.4f}  flip={flip:.4f}",
              flush=True)
    return out


# ─── 2. Threshold MIAs ────────────────────────────────────────────────────────

def _per_sample(model, cache, idx, labels, device, probe_eps, full):
    bce_none, bce = nn.BCEWithLogitsLoss(reduction="none"), nn.BCEWithLogitsLoss()
    keys = ["loss", "certainty", "adv_loss"] + (["adv_loss_pgd", "paper_fluct"] if full else []) \
        + [f"fluct_e{e:g}" for e in FLUCT_EPS] + ["dlogit_e4", "probs", "labels"]
    out = {k: [] for k in keys}
    gen = v2.make_generator(device, 5)
    for x, y in v2.loader(cache, idx, labels, batch_size=128, workers=6, device=device):
        x = x.to(device, non_blocking=True).contiguous(memory_format=v2.CHANNELS_LAST)
        y = y.to(device, non_blocking=True)
        with torch.no_grad(), v2.amp_ctx(device):
            z = model(x).float()
        p = torch.sigmoid(z)
        own = (p > 0.5).float()
        x_adv = v2.fgsm(model, x, y, probe_eps, bce)
        with torch.no_grad(), v2.amp_ctx(device):
            z_adv = model(x_adv).float()
        out["loss"].append(bce_none(z, y).mean(1).cpu().numpy())
        out["certainty"].append((2 * p - 1).abs().mean(1).cpu().numpy())
        out["adv_loss"].append(bce_none(z_adv, y).mean(1).cpu().numpy())
        if full:
            x_pgd = v2.pgd(model, x, y, probe_eps, THRESH_PGD_STEPS, bce, generator=gen)
            with torch.no_grad(), v2.amp_ctx(device):
                z_pgd = model(x_pgd).float()
            out["adv_loss_pgd"].append(bce_none(z_pgd, y).mean(1).cpu().numpy())
            # paper Eq. 7: per-class |Δp| under FGSM (ε = 0.01 normalised, true labels, clamp ±3)
            x_pf = v2.fgsm_v1(x, v2.input_grad(model, x, y, bce).sign(), PAPER_FLUCT_EPS)
            with torch.no_grad(), v2.amp_ctx(device):
                p_pf = torch.sigmoid(model(x_pf).float())
            out["paper_fluct"].append((p - p_pf).abs().cpu().numpy())
        for e in FLUCT_EPS:
            x_p = v2.fgsm(model, x, own, e, bce)
            with torch.no_grad(), v2.amp_ctx(device):
                z_p = model(x_p).float()
            out[f"fluct_e{e:g}"].append((p - torch.sigmoid(z_p)).abs().sum(1).cpu().numpy())
            if e == 4.0:
                out["dlogit_e4"].append((z - z_p).abs().mean(1).cpu().numpy())
        out["probs"].append(p.cpu().numpy())
        out["labels"].append(y.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items() if v}


SCORE_SIGN = {"loss": -1, "certainty": +1, "adv_loss": -1, "adv_loss_pgd": -1,
              "fluct_e1": -1, "fluct_e4": -1, "dlogit_e4": -1}


def threshold_eval(model, cache, device, m_idx, nm_idx, y_m, y_nm, probe_eps, full):
    m = _per_sample(model, cache, m_idx, y_m, device, probe_eps, full)
    n = _per_sample(model, cache, nm_idx, y_nm, device, probe_eps, full)
    gen = {"member_loss": float(m["loss"].mean()), "nonmember_loss": float(n["loss"].mean()),
           "loss_gap": float(n["loss"].mean() - m["loss"].mean()),
           "member_auc": mc.macro_auc(m["labels"], m["probs"]),
           "nonmember_auc": mc.macro_auc(n["labels"], n["probs"]),
           "adv_loss_gap": float(n["adv_loss"].mean() - m["adv_loss"].mean())}
    gen["auc_gap"] = gen["member_auc"] - gen["nonmember_auc"]
    for e in FLUCT_EPS:
        k = f"fluct_e{e:g}"
        gen[f"member_{k}"], gen[f"nonmember_{k}"] = float(m[k].mean()), float(n[k].mean())
    # paper text metrics: average max confidence and Eq. 7 average per-class fluctuation
    gen["avg_max_conf_members"] = float(m["probs"].max(1).mean())
    gen["avg_max_conf_nonmembers"] = float(n["probs"].max(1).mean())
    if "paper_fluct" in m:
        pm, pn = m["paper_fluct"].mean(0), n["paper_fluct"].mean(0)
        gen["paper_fluct"] = {
            "eps_normalised": PAPER_FLUCT_EPS, "labels": "true", "gradient_mode": "eval",
            "avg_members_pct": float(100 * pm.mean()), "avg_nonmembers_pct": float(100 * pn.mean()),
            "per_class_members": [float(v) for v in pm],
            "per_class_nonmembers": [float(v) for v in pn],
            "class_names": v2.DISEASE_CLASSES}
    y = np.concatenate([np.ones(len(m_idx)), np.zeros(len(nm_idx))])
    attacks, scores = {}, {}
    for k, sign in SCORE_SIGN.items():
        if k not in m:
            continue
        s = sign * np.concatenate([m[k], n[k]])
        attacks[k] = v2.score_metrics(y, s)
        scores[k] = s.astype(np.float32)
    return {"generalization": gen, "threshold_attacks": attacks}, scores


def main():
    p = argparse.ArgumentParser(description="v2 evasion + threshold-MIA evaluation of one victim")
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=v2.DEFAULT_CACHE_DIR)
    p.add_argument("--probe_eps", type=float, default=DEFAULT_PROBE,
                   help="ε/255 of the adversarial-loss attack (same for every victim)")
    p.add_argument("--skip_checkpoints", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    device = v2.get_device()
    torch.backends.cudnn.benchmark = True
    cache = mc.ImageCache(args.cache_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    meta = v2.read_json(os.path.join(args.victim_dir, "model_meta.json"), {})
    t0 = time.time()
    model = v2.load_victim(os.path.join(args.victim_dir, "model.pth"), device)

    ev_path = os.path.join(args.out_dir, "evasion.json")
    if os.path.exists(ev_path):
        print("[EVAL] evasion.json exists — skipping evasion", flush=True)
    else:
        print(f"[EVAL] {args.victim_dir}: evasion …", flush=True)
        ev = run_evasion(model, cache, device, args.smoke)
        ev["victim_dir"] = args.victim_dir
        v2.write_json(ev_path, ev)

    _, eval_m, eval_nm = v2.attack_split(args.seed, smoke=args.smoke)
    m_idx, nm_idx = cache.index_of(eval_m), cache.index_of(eval_nm)
    y_m, y_nm = v2.labels_for_paths(eval_m), v2.labels_for_paths(eval_nm)
    part_path = os.path.join(args.out_dir, "threshold_partial.json")
    part = v2.read_json(part_path, {"final": None, "checkpoints": {}})

    if part["final"] is None:
        print(f"[EVAL] threshold MIAs on {len(m_idx)} + {len(nm_idx)} images (final model) …",
              flush=True)
        part["final"], scores = threshold_eval(model, cache, device, m_idx, nm_idx, y_m, y_nm,
                                               args.probe_eps, full=True)
        v2.save_npz(os.path.join(args.out_dir, "threshold_scores.npz"),
                    y_true=np.concatenate([np.ones(len(m_idx)), np.zeros(len(nm_idx))]).astype(np.int8),
                    **scores)
        v2.write_json(part_path, part)
    fin = part["final"]["threshold_attacks"]
    print("  final: " + "  ".join(f"{k}_AUC={v['auc']:.4f}" for k, v in fin.items()), flush=True)

    if not args.skip_checkpoints:
        for ck in sorted(glob.glob(os.path.join(args.victim_dir, "checkpoints", "epoch_*.pth"))):
            epoch = int(re.search(r"epoch_(\d+)", os.path.basename(ck)).group(1))
            if str(epoch) in part["checkpoints"]:
                continue
            m = v2.load_victim(ck, device)
            r, _ = threshold_eval(m, cache, device, m_idx, nm_idx, y_m, y_nm, args.probe_eps,
                                  full=False)
            r["epoch"] = epoch
            part["checkpoints"][str(epoch)] = r
            v2.write_json(part_path, part)
            a = r["threshold_attacks"]
            print(f"  epoch {epoch:02d}: loss_gap={r['generalization']['loss_gap']:+.4f} "
                  f"loss_AUC={a['loss']['auc']:.4f} adv_loss_AUC={a['adv_loss']['auc']:.4f} "
                  f"fluct_e4_AUC={a['fluct_e4']['auc']:.4f}", flush=True)
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {"victim_dir": args.victim_dir, "seed": args.seed, "probe_eps255": args.probe_eps,
              "selected_epoch": meta.get("selected_epoch"), "final": part["final"],
              "checkpoints": [part["checkpoints"][k]
                              for k in sorted(part["checkpoints"], key=int)],
              "runtime_s": round(time.time() - t0, 1)}
    v2.write_json(os.path.join(args.out_dir, "threshold_mia.json"), result)
    print(f"[EVAL] done in {result['runtime_s']:.0f}s → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
