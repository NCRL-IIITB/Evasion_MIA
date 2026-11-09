"""
run_attacks.py
==============
The two black-box membership inference attacks of the comparison, run on one
victim with the shadow sets trained by train_shadows.py.

--attack shadow_mia --shadow_kind std   "Shadow-MIA": the paper's shadow-model
        MIA (Shokri et al.) — 15 victim probabilities + max + argmax.
--attack shadow_mia --shadow_kind at    "Shadow-MIA (AT shadows)": same
        features, shadows adversarially trained.  A defence-AWARE attacker
        (Song et al. 2019 assume the attacker knows the defence) — reported
        separately because it favours the FGSM victim by construction.
--attack afmia                          AFMIA, three feature sets:
  "AFMIA (paper features)"   v1 recipe: 15 per-class |Δp| + max + argmax; shadow
                             rows probed with the shadow's own gradient, the
                             victim probed by transfer from shadow 0, ε = 0.01
                             normalised, clamp ±3.
  "AFMIA-v2 (perturbation only)"  ONLY the response to perturbation, no clean
                             confidence: for ε ∈ {1,2,4,8}/255 the sum / max of
                             |Δp|, mean / max of |Δlogit|, number of label flips,
                             plus per-class |Δlogit| at 2/255.  This isolates the
                             robust-overfitting signal that the paper attributes
                             to adversarial training.
  "AFMIA-v2 (+ clean confidence)" the above + max conf, argmax, certainty,
                             clean loss vs own prediction, adversarial loss vs
                             the clean prediction at every ε (Song et al.).

How the v2 perturbations are built (the fixes)
---------------------------------------------
* Ensemble transfer: the probe direction is sign(Σ_k ĝ_k), ĝ_k = the
  L1-normalised input gradient of shadow k w.r.t. its own predictions.  v1
  transferred from one ResNet-18.
* Train = test: shadow rows are probed exactly like the victim — by transfer
  from OTHER shadows that never trained on the image (at attack time no shadow
  has seen the evaluation images).  v1 probed shadow rows with their own
  white-box gradient but the victim by transfer, so the attack classifier
  was trained on a different feature distribution than it was applied to.
* Several ε and logit-space features: members of a memorising model sit at
  p ≈ 0 / 1, where |Δp| is ~0 whatever the robustness; |Δlogit| is not.

Attack dataset rows: a fixed random subset A of the pool (10,000 images) ×
8 shadows; label = whether the image was in that shadow's training set;
balanced per shadow by subsampling non-members (≈ 40k rows).

Outputs (<out_dir>):
  results.json / results.txt   metrics for every feature set × classifier:
                               AUC, TPR@1%/0.1% FPR, acc@0.5, acc@median
                               (prior-matched, label-free), best-threshold acc
                               (optimistic), precision / recall / F1
  scores.npz                   per-sample membership scores on the 5k + 5k
                               evaluation images (members first) — for the
                               paired C-vs-D bootstrap after merging
  _cache/*.npz                 shadow-side and victim-side features (resume,
                               and re-analysis without a GPU)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as v2  # noqa: E402

mc = v2.mc

N_A          = 10_000
PROBE_EPS    = [1.0, 2.0, 4.0, 8.0]
REF_EPS      = 2.0
V1_EPS_NORM  = 0.01
BATCH        = 32

FS_SHADOW     = "Shadow-MIA"                       # paper: "Original MIA"
FS_SHADOW_AT  = "Shadow-MIA (AT shadows)"
FS_AF_PAPER   = "AFMIA (paper features)"           # paper: "Fluctuation"
FS_AF_PERT    = "AFMIA-v2 (perturbation only)"
FS_AF_CONF    = "AFMIA-v2 (+ clean confidence)"
BCVA          = " + BCVA"                          # paper: "+ Variance"
BOOTSTRAP_K   = 100                                # v1 mia_variance.BOOTSTRAP_K


def _bootstrap_var(pool, n, rng, k=BOOTSTRAP_K):
    return np.var(pool[rng.randint(0, len(pool), size=(n, k))], axis=1)


def bcva_columns(rk, row_maxconf, victim_pool_maxconf, n_eval, seed):
    """The paper's variance-of-max feature (v1 mia_variance / mia_fluctuation_variance):
    each shadow row gets var() of K=100 values bootstrapped from THAT shadow's clean
    max-confidence pool (its attack rows); each victim row gets var() bootstrapped
    from the victim's max confidences on all shadow training images."""
    rng = np.random.RandomState(seed + 1)
    tr = np.zeros(len(rk))
    for k in np.unique(rk):
        sel = rk == k
        tr[sel] = _bootstrap_var(row_maxconf[sel], int(sel.sum()), rng)
    ev = _bootstrap_var(victim_pool_maxconf, n_eval, rng)
    return tr[:, None], ev[:, None]


def victim_pool_maxconf(plan, vprobs):
    """Victim max confidence on every shadow's training images, concatenated (as v1)."""
    return np.concatenate([vprobs[np.asarray(s["train_indices"])].max(1) for s in plan["shadows"]])


def classifiers(seed):
    """Same four classifiers as the paper; Gradient Boosting is sklearn's
    histogram implementation (same model family, minutes instead of hours)."""
    return [
        ("Gradient Boosting", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                             random_state=seed)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)),
        ("MLP", make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                                            early_stopping=True, random_state=seed))),
        ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ]


# ─── Shadows, rows ────────────────────────────────────────────────────────────

def load_shadow_set(attack_dir, kind, device):
    d = os.path.join(attack_dir, f"shadows_{kind}")
    if not os.path.exists(os.path.join(d, "DONE.json")):
        sys.exit(f"ERROR: shadow set '{kind}' is not complete in {d} — run train_shadows.py first")
    plan = v2.read_json(os.path.join(d, "plan.json"))
    shadows = [v2.load_shadow(s["arch"], os.path.join(d, f"shadow_{s['i']}.pth"), device)
               for s in plan["shadows"]]
    for m in shadows:
        for prm in m.parameters():
            prm.requires_grad_(False)
    return plan, shadows


def attack_subset(plan, seed, n_a):
    n_pool = plan["pool_size"]
    rng = np.random.RandomState(seed + 2)
    A = rng.choice(n_pool, size=min(n_a, n_pool), replace=False)
    M = np.zeros((len(A), len(plan["shadows"])), dtype=bool)
    for k, s in enumerate(plan["shadows"]):
        M[:, k] = np.isin(A, np.asarray(s["train_indices"]))
    return A, M


def balanced_rows(M, seed):
    """(a, k, label) rows, members and non-members balanced per shadow."""
    rng = np.random.RandomState(seed + 3)
    rows = []
    for k in range(M.shape[1]):
        mem, non = np.where(M[:, k])[0], np.where(~M[:, k])[0]
        n = min(len(mem), len(non))
        if n == 0:
            continue
        mem = rng.choice(mem, n, replace=False) if len(mem) > n else mem
        non = rng.choice(non, n, replace=False)
        rows += [(a, k, 1) for a in mem] + [(a, k, 0) for a in non]
    return np.asarray(rows, dtype=np.int64)


# ─── Feature builders (numpy) ─────────────────────────────────────────────────

def paper_conf_features(P):
    return np.hstack([P, P.max(1, keepdims=True), P.argmax(1)[:, None].astype(np.float64)])


def paper_afmia_features(P0, PV1):
    return np.hstack([np.abs(P0 - PV1), P0.max(1, keepdims=True),
                      P0.argmax(1)[:, None].astype(np.float64)])


def v2_afmia_features(P0, PA):
    """P0 (N,15) clean probs, PA (N,E,15) probs under the probes → (pert, conf) matrices."""
    Z0 = v2.logit_np(P0)
    yhat = (P0 > 0.5).astype(np.float64)
    pert, pert_names = [], []
    for j, e in enumerate(PROBE_EPS):
        Pa = PA[:, j]
        d, dz = np.abs(P0 - Pa), np.abs(Z0 - v2.logit_np(Pa))
        pert += [d.sum(1), d.max(1), dz.mean(1), dz.max(1), ((Pa > 0.5) != (P0 > 0.5)).sum(1)]
        pert_names += [f"fl_sum_e{e:g}", f"fl_max_e{e:g}", f"dz_mean_e{e:g}", f"dz_max_e{e:g}",
                       f"flips_e{e:g}"]
    j_ref = PROBE_EPS.index(REF_EPS)
    dz_ref = np.abs(Z0 - v2.logit_np(PA[:, j_ref]))
    pert += list(dz_ref.T)
    pert_names += [f"dz_class{c}_e{REF_EPS:g}" for c in range(dz_ref.shape[1])]
    conf = [P0.max(1), P0.argmax(1).astype(np.float64), np.abs(2 * P0 - 1).mean(1),
            v2.bce_np(P0, yhat).mean(1)]
    conf_names = ["max_conf", "max_class", "certainty", "clean_loss_own"]
    for j, e in enumerate(PROBE_EPS):
        conf.append(v2.bce_np(PA[:, j], yhat).mean(1))
        conf_names.append(f"adv_loss_own_e{e:g}")
    return (np.column_stack(pert), pert_names, np.column_stack(conf), conf_names)


def threshold_scores(P0, PA):
    """Classifier-free scores (higher = member) for diagnostics."""
    Z0, yhat = v2.logit_np(P0), (P0 > 0.5).astype(np.float64)
    out = {"certainty": np.abs(2 * P0 - 1).mean(1), "max_conf": P0.max(1),
           "neg_clean_loss_own": -v2.bce_np(P0, yhat).mean(1)}
    for j, e in enumerate(PROBE_EPS):
        Pa = PA[:, j]
        out[f"neg_fl_sum_e{e:g}"] = -np.abs(P0 - Pa).sum(1)
        out[f"neg_dz_mean_e{e:g}"] = -np.abs(Z0 - v2.logit_np(Pa)).mean(1)
        out[f"neg_adv_loss_own_e{e:g}"] = -v2.bce_np(Pa, yhat).mean(1)
    return out


# ─── GPU passes ───────────────────────────────────────────────────────────────

def _normalised_grad(model, x, crit):
    """L1-normalised input gradient w.r.t. the model's own thresholded prediction.
    Returns (ĝ, own labels, clean fp32 probabilities)."""
    with torch.no_grad():
        p = torch.sigmoid(model(x).float())
    own = (p > 0.5).float()
    g = v2.input_grad(model, x, own, crit, amp=True)
    return g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-20), own, p


@torch.no_grad()
def _probs(model, x, channels_last=False):
    if channels_last:
        x = x.contiguous(memory_format=v2.CHANNELS_LAST)
    return torch.sigmoid(model(x).float())


def shadow_side_afmia(shadows, cache, img_idx, M, victim_pseudo, device, log):
    """Clean / v1-probe / v2-probe probabilities of every shadow on every image of A."""
    N, K, E, C = len(img_idx), len(shadows), len(PROBE_EPS), v2.NUM_CLASSES
    P0 = np.zeros((N, K, C), np.float32)
    PV1 = np.zeros((N, K, C), np.float32)
    PA = np.zeros((N, K, E, C), np.float32)
    crit = nn.BCEWithLogitsLoss()
    pos, t0 = 0, time.time()
    for x in v2.loader(cache, img_idx, batch_size=BATCH, workers=4, device=device):
        x = x.to(device, non_blocking=True)
        b = slice(pos, pos + len(x))
        Mb = torch.from_numpy(M[b]).to(device)
        yv = torch.from_numpy(victim_pseudo[b]).to(device)
        G = []
        for k, f in enumerate(shadows):
            g, own, p_clean = _normalised_grad(f, x, crit)
            G.append(g)
            P0[b, k] = p_clean.cpu().numpy()
            # v1 probe: own white-box gradient; labels = victim pseudo-labels for the
            # shadow's members, the shadow's own predictions otherwise (as v1)
            lab = torch.where(Mb[:, k:k + 1], yv, own)
            g1 = g if torch.equal(lab, own) else v2.input_grad(f, x, lab, crit, amp=True)
            PV1[b, k] = _probs(f, v2.fgsm_v1(x, g1.sign(), V1_EPS_NORM)).cpu().numpy()
        G = torch.stack(G)                                       # (K, B, 3, H, W)
        for i, f in enumerate(shadows):
            w = (~Mb).float()                                    # sources that never saw x
            w[:, i] = 0.0
            empty = w.sum(1) == 0
            if empty.any():                                      # (rare) fall back to all others
                w[empty] = 1.0
                w[empty, i] = 0.0
            d = torch.einsum("bk,kbchw->bchw", w, G).sign()
            for j, e in enumerate(PROBE_EPS):
                PA[b, i, j] = _probs(f, v2.clamp_valid(x + v2.eps_t(e, x) * d)).cpu().numpy()
        del G
        pos += len(x)
        if pos % (BATCH * 50) < BATCH:
            log(f"    shadow side {pos}/{N} images ({(time.time() - t0) / 60:.1f} min)")
    return P0, PV1, PA


def victim_side_afmia(victim, shadows, cache, img_idx, device, log):
    N, E, C = len(img_idx), len(PROBE_EPS), v2.NUM_CLASSES
    P0 = np.zeros((N, C), np.float32)
    PV1 = np.zeros((N, C), np.float32)
    PA = np.zeros((N, E, C), np.float32)
    crit = nn.BCEWithLogitsLoss()
    pos, t0 = 0, time.time()
    for x in v2.loader(cache, img_idx, batch_size=BATCH, workers=4, device=device):
        x = x.to(device, non_blocking=True)
        b = slice(pos, pos + len(x))
        pv = _probs(victim, x, channels_last=True)
        P0[b] = pv.cpu().numpy()
        yv = (pv > 0.5).float()
        g0 = v2.input_grad(shadows[0], x, yv, crit, amp=True)        # v1: transfer from shadow 0
        PV1[b] = _probs(victim, v2.fgsm_v1(x, g0.sign(), V1_EPS_NORM), channels_last=True).cpu().numpy()
        S = torch.zeros_like(x)
        for f in shadows:                                            # v2: ensemble transfer
            S += _normalised_grad(f, x, crit)[0]
        del g0
        d = S.sign()
        for j, e in enumerate(PROBE_EPS):
            PA[b, j] = _probs(victim, v2.clamp_valid(x + v2.eps_t(e, x) * d),
                              channels_last=True).cpu().numpy()
        pos += len(x)
        if pos % (BATCH * 50) < BATCH:
            log(f"    victim side {pos}/{N} images ({(time.time() - t0) / 60:.1f} min)")
    return P0, PV1, PA


def shadow_side_conf(shadows, cache, img_idx, device):
    return np.stack([v2.predict_probs(f, cache, img_idx, device, batch_size=128, amp=False)
                     for f in shadows], axis=1)                      # (N, K, 15)


# ─── Classifiers ──────────────────────────────────────────────────────────────

def _clean(X):
    return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)


def evaluate_feature_set(name, X_train, y_train, X_eval, y_eval, seed, log):
    rows, scores = [], {}
    X_train, X_eval = _clean(X_train), _clean(X_eval)
    for label, clf in classifiers(seed):
        t0 = time.time()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_eval)
        s = clf.predict_proba(X_eval)[:, 1]
        m = v2.score_metrics(y_eval, s)
        r = {"feature_set": name, "classifier": label, "attack_label": f"{name} / {label}",
             "accuracy": float(accuracy_score(y_eval, pred)),
             "precision": float(precision_score(y_eval, pred, zero_division=0)),
             "recall": float(recall_score(y_eval, pred, zero_division=0)),
             "f1": float(f1_score(y_eval, pred, zero_division=0)),
             "auc": m["auc"], "tpr_at_1pct_fpr": m["tpr_at_1pct_fpr"],
             "tpr_at_0.1pct_fpr": m["tpr_at_0.1pct_fpr"], "acc_median": m["acc_median"],
             "best_threshold_acc": m["best_threshold_acc"],
             "n_train_rows": int(len(y_train)), "n_features": int(X_train.shape[1]),
             "seconds": round(time.time() - t0, 1)}
        rows.append(r)
        scores[f"{name}|{label}"] = s.astype(np.float32)
        log(f"  {r['attack_label']:<52s} AUC={r['auc']:.4f} TPR@1%={r['tpr_at_1pct_fpr']:.4f} "
            f"acc@0.5={r['accuracy']:.4f} acc@median={r['acc_median']:.4f}")
    return rows, scores


def write_txt(path, title, rows, thr_rows):
    with open(path, "w") as f:
        f.write(f"{title}\n{'=' * 120}\n")
        f.write(f"{'Attack':<56s}{'AUC':>8}{'TPR@1%':>9}{'TPR@.1%':>9}{'Acc@.5':>8}"
                f"{'Acc@med':>9}{'Prec':>8}{'Rec':>8}{'F1':>8}\n{'-' * 128}\n")
        for r in rows:
            f.write(f"{r['attack_label']:<56s}{r['auc']:8.4f}{r['tpr_at_1pct_fpr']:9.4f}"
                    f"{r['tpr_at_0.1pct_fpr']:9.4f}{r['accuracy']:8.4f}{r['acc_median']:9.4f}"
                    f"{r['precision']:8.4f}{r['recall']:8.4f}{r['f1']:8.4f}\n")
        if thr_rows:
            f.write(f"\nClassifier-free scores on the victim (AUC; shadow-row AUC = same score "
                    f"on the attacker's shadow rows)\n{'-' * 120}\n")
            for r in thr_rows:
                f.write(f"{r['name']:<56s}{r['auc']:8.4f}{r['tpr_at_1pct_fpr']:9.4f}"
                        f"{'':9s}{'':8s}{r['acc_median']:9.4f}   shadow-row AUC "
                        f"{r['shadow_row_auc']:.4f}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="v2 black-box MIAs on one victim")
    p.add_argument("--attack", required=True, choices=["shadow_mia", "afmia"])
    p.add_argument("--shadow_kind", default=None, choices=["std", "at"],
                   help="shadow_mia only (afmia always uses the AT shadows)")
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--attack_dir", required=True, help="results_v2/attacks/<run>")
    p.add_argument("--cache_dir", default=v2.DEFAULT_CACHE_DIR)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    kind = "at" if args.attack == "afmia" else (args.shadow_kind or "std")
    name = "afmia" if args.attack == "afmia" else f"shadow_mia_{kind}"
    out_dir = os.path.join(args.attack_dir, name)
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    device = v2.get_device()
    torch.backends.cudnn.benchmark = True
    t_start = time.time()
    cache = mc.ImageCache(args.cache_dir)
    pool, eval_m, eval_nm = v2.attack_split(args.seed, smoke=args.smoke)
    pool_idx = cache.index_of(pool)
    eval_idx = np.concatenate([cache.index_of(eval_m), cache.index_of(eval_nm)])
    y_eval = np.concatenate([np.ones(len(eval_m)), np.zeros(len(eval_nm))]).astype(int)
    meta = v2.read_json(os.path.join(args.victim_dir, "model_meta.json"), {})

    plan, shadows = load_shadow_set(args.attack_dir, kind, device)
    A, M = attack_subset(plan, args.seed, 200 if args.smoke else N_A)
    rows = balanced_rows(M, args.seed)
    ra, rk, ry = rows[:, 0], rows[:, 1], rows[:, 2]
    log(f"[{name}] victim {args.victim_dir}: {len(shadows)} {kind} shadows, attack subset "
        f"{len(A)} images → {len(rows)} balanced rows; eval {len(eval_m)} + {len(eval_nm)}")
    victim = v2.load_victim(os.path.join(args.victim_dir, "model.pth"), device)
    for prm in victim.parameters():
        prm.requires_grad_(False)

    results, all_scores, thr_rows = [], {}, []

    if args.attack == "shadow_mia":
        sp = os.path.join(cache_dir, "features.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            P_sh, P_v = z["P_shadow"], z["P_victim"]
            log("  features loaded from cache")
        else:
            P_sh = shadow_side_conf(shadows, cache, pool_idx[A], device)
            P_v = v2.predict_probs(victim, cache, eval_idx, device, amp=False, channels_last=True)
            v2.save_npz(sp, P_shadow=P_sh, P_victim=P_v, A=A, M=M)
        fs = FS_SHADOW if kind == "std" else FS_SHADOW_AT
        X_tr = paper_conf_features(P_sh[ra, rk].astype(np.float64))
        X_ev = paper_conf_features(P_v.astype(np.float64))
        sets = [(fs, X_tr, X_ev)]
        if kind == "std":                                  # paper: "Original MIA + Variance"
            vprobs = v2.victim_pool_probs(args.attack_dir, args.victim_dir, cache, pool_idx, device)
            b_tr, b_ev = bcva_columns(rk, P_sh[ra, rk].max(1), victim_pool_maxconf(plan, vprobs),
                                      len(y_eval), args.seed)
            sets.append((fs + BCVA, np.hstack([X_tr, b_tr]), np.hstack([X_ev, b_ev])))
        for name_fs, a, b in sets:
            r, s = evaluate_feature_set(name_fs, a, ry, b, y_eval, args.seed, log)
            results += r
            all_scores.update(s)
        sh_scores = {"certainty": np.abs(2 * P_sh[ra, rk] - 1).mean(1),
                     "max_conf": P_sh[ra, rk].max(1)}
        v_scores = {"certainty": np.abs(2 * P_v - 1).mean(1), "max_conf": P_v.max(1)}
    else:
        sp, vp = os.path.join(cache_dir, "shadow_side.npz"), os.path.join(cache_dir, "victim_side.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            P0s, PV1s, PAs = z["P0"], z["PV1"], z["PA"]
            log("  shadow-side features loaded from cache")
        else:
            vprobs = v2.victim_pool_probs(args.attack_dir, args.victim_dir, cache, pool_idx, device)
            pseudo_A = (vprobs[A] > 0.5).astype(np.float32)
            log("  shadow side: clean, v1 and v2 probes on the attack subset …")
            P0s, PV1s, PAs = shadow_side_afmia(shadows, cache, pool_idx[A], M, pseudo_A, device, log)
            v2.save_npz(sp, P0=P0s, PV1=PV1s, PA=PAs, A=A, M=M)
        if os.path.exists(vp):
            z = np.load(vp)
            P0v, PV1v, PAv = z["P0"], z["PV1"], z["PA"]
            log("  victim-side features loaded from cache")
        else:
            log("  victim side: clean, v1 and v2 probes on the evaluation images …")
            P0v, PV1v, PAv = victim_side_afmia(victim, shadows, cache, eval_idx, device, log)
            v2.save_npz(vp, P0=P0v, PV1=PV1v, PA=PAv)

        P0r, PV1r, PAr = (P0s[ra, rk].astype(np.float64), PV1s[ra, rk].astype(np.float64),
                          PAs[ra, rk].astype(np.float64))
        P0v, PV1v, PAv = P0v.astype(np.float64), PV1v.astype(np.float64), PAv.astype(np.float64)
        pert_tr, _, conf_tr, _ = v2_afmia_features(P0r, PAr)
        pert_ev, pert_names, conf_ev, conf_names = v2_afmia_features(P0v, PAv)
        vprobs = v2.victim_pool_probs(args.attack_dir, args.victim_dir, cache, pool_idx, device)
        b_tr, b_ev = bcva_columns(rk, P0r.max(1), victim_pool_maxconf(plan, vprobs),
                                  len(y_eval), args.seed)
        paper_tr, paper_ev = paper_afmia_features(P0r, PV1r), paper_afmia_features(P0v, PV1v)
        for fs, X_tr, X_ev in (
                (FS_AF_PAPER, paper_tr, paper_ev),
                (FS_AF_PAPER + BCVA, np.hstack([paper_tr, b_tr]), np.hstack([paper_ev, b_ev])),
                (FS_AF_PERT, pert_tr, pert_ev),
                (FS_AF_CONF, np.hstack([pert_tr, conf_tr]), np.hstack([pert_ev, conf_ev]))):
            r, s = evaluate_feature_set(fs, X_tr, ry, X_ev, y_eval, args.seed, log)
            results += r
            all_scores.update(s)
        sh_scores, v_scores = threshold_scores(P0r, PAr), threshold_scores(P0v, PAv)
        diag = {"feature_names": {"perturbation": pert_names, "confidence": conf_names}}
        for j, e in enumerate(PROBE_EPS):
            d = np.abs(P0v - PAv[:, j]).sum(1)
            diag[f"victim_mean_fl_sum_e{e:g}"] = {"members": float(d[y_eval == 1].mean()),
                                                  "nonmembers": float(d[y_eval == 0].mean())}

    for k in v_scores:
        m = v2.score_metrics(y_eval, v_scores[k])
        sh_auc = v2.score_metrics(ry, sh_scores[k])["auc"]
        thr_rows.append({"name": f"threshold: {k}", "key": k, **m, "shadow_row_auc": sh_auc})
        all_scores[f"threshold|{k}"] = np.asarray(v_scores[k], dtype=np.float32)

    conf_gap = None
    if args.attack == "shadow_mia":
        conf_gap = float(P_v[y_eval == 1].max(1).mean() - P_v[y_eval == 0].max(1).mean())
    out = {"attack": name, "victim_dir": args.victim_dir, "seed": args.seed,
           "shadow_kind": kind, "conf_gap": conf_gap,
           "config": {"n_shadows": len(shadows), "attack_subset": int(len(A)),
                      "n_rows": int(len(rows)), "probe_eps255": PROBE_EPS,
                      "ref_eps255": REF_EPS, "v1_eps_normalised": V1_EPS_NORM,
                      "eval_members": int(len(eval_m)), "eval_nonmembers": int(len(eval_nm)),
                      "shadow_plan": {k: v for k, v in plan.items() if k != "shadows"},
                      "member_definition": "victim training images only (validation excluded)"},
           "results": results, "threshold_scores": thr_rows,
           "diagnostics": diag if args.attack == "afmia" else {},
           "runtime_s": round(time.time() - t_start, 1),
           "victim_meta": {k: v for k, v in meta.items() if k != "label_names"}}
    v2.save_npz(os.path.join(out_dir, "scores.npz"), y_true=y_eval.astype(np.int8), **all_scores)
    v2.write_json(os.path.join(out_dir, "results.json"), out)
    write_txt(os.path.join(out_dir, "results.txt"),
              f"{name} on {args.victim_dir} (seed {args.seed})", results, thr_rows)
    log(f"[DONE] {name} → {out_dir} ({out['runtime_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
