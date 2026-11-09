"""
lab_attacks.py
==============
Experimental black-box membership inference attacks ("the lab"), run on one
victim with the two shadow sets that Matched_Experiments_v2/train_shadows.py
trained and kept (standard shadows and FGSM-RS adversarially trained shadows).
Uses the SAME attack-dataset rows as v2's run_attacks.py (10k pool images × 8
shadows, balanced) and the same 5k + 5k evaluation images, so every number is
directly comparable with the paper-faithful and v2 attacks.

Feature sets (every one is trained on shadow rows and applied to the victim):

  LiRA-cal confidence      (standard shadows)  per-class margin m_c =
        (2ŷ_c − 1)·logit(p_c) w.r.t. the target's own prediction ŷ, plus the same
        margin of REFERENCE models that never trained on the image (shadows
        k ≠ i not trained on it; for the victim: all shadows), as m − μ_ref and
        (m − μ_ref)/σ_ref.  Offline-LiRA-style difficulty calibration (Carlini et
        al., S&P 2022): an image the victim is very sure about is only
        suspicious if models that did NOT train on it are unsure.
  Robustness-cal (perturbation only)  (AT shadows)  v2's perturbation-response
        features (ensemble-transfer probes at 1/2/4/8 per 255, |Δp|, |Δlogit|,
        flips) raw AND minus the same features of reference models.  Isolates
        "the victim is unusually stable on THIS image" — the robust-overfitting
        signature — from "this image is stable for every model".
  Robustness-cal (+ clean confidence)  the above + calibrated confidence.
  Query robustness (Square + noise)  (AT shadows)  NO gradients and NO transfer:
        a score-based L∞ Square attack (Andriushchenko et al., ECCV 2020) at
        ε = 4/255 with 60 queries on the loss w.r.t. the model's own prediction
        (loss after 1/10/30/60 queries), plus random ±ε sign-noise stability at
        8/255 (4 draws); raw and reference-calibrated.  Measures local
        robustness of the victim directly through its API.
  All black-box (combined)  every block above for the same (image, shadow) rows.

Classifier-free scores (fixed direction: more confident / more stable =
member) are reported too, with their AUC on the attacker's shadow rows.

Outputs (<attack_dir>/lab/): results.json, results.txt, scores.npz (same schema
as v2's run_attacks.py) and _cache/*.npz (all raw probe outputs).
"""

import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE   = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(os.path.dirname(HERE), "Matched_Experiments_v2")
sys.path.insert(0, V2_DIR)
import v2_common as v2          # noqa: E402

# v2's run_attacks.py by file path (never a same-named file elsewhere on sys.path)
_spec = importlib.util.spec_from_file_location("v2_run_attacks", os.path.join(V2_DIR, "run_attacks.py"))
v2a = importlib.util.module_from_spec(_spec)   # v2 rows, classifiers, feature builders
_spec.loader.exec_module(v2a)

mc = v2.mc

PROBE_EPS   = v2a.PROBE_EPS            # [1, 2, 4, 8] / 255
NOISE_EPS   = 8.0
NOISE_DRAWS = 4
SQ_EPS      = 4.0
SQ_QUERIES  = 60
SQ_CHECK    = [1, 10, 30, 60]
SQ_PINIT    = 0.1
N_A         = v2a.N_A                  # 10,000 (same rows as v2)
N_Q         = 3000                     # images of A that also get the Square probe
BATCH       = 32
SIGMA_FLOOR = 0.25                     # logit units

FS_LIRA     = "LiRA-cal confidence"
FS_ROB      = "Robustness-cal (perturbation only)"
FS_ROB_CONF = "Robustness-cal (+ clean confidence)"
FS_QUERY    = "Query robustness (Square + noise)"
FS_ALL      = "All black-box (combined)"


# ─── Probes (torch) ───────────────────────────────────────────────────────────

@torch.no_grad()
def probs_of(model, x, amp=False, cl=False):
    if cl:
        x = x.contiguous(memory_format=v2.CHANNELS_LAST)
    with v2.amp_ctx(x.device, amp):
        z = model(x)
    return torch.sigmoid(z.float())


def own_bce(p, y):
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(y * torch.log(p) + (1 - y) * torch.log1p(-p)).mean(1)


@torch.no_grad()
def noise_probe(model, x, p0, gen, cl=False):
    """Mean Σ|Δp| and mean |Δlogit| under random ±ε sign noise (ε = 8/255)."""
    e = v2.eps_t(NOISE_EPS, x)
    z0 = torch.logit(p0.clamp(1e-7, 1 - 1e-7))
    fl = torch.zeros(len(x), device=x.device)
    dz = torch.zeros(len(x), device=x.device)
    for _ in range(NOISE_DRAWS):
        s = (torch.rand(x.shape, device=x.device, generator=gen) < 0.5).float() * 2 - 1
        pn = probs_of(model, v2.clamp_valid(x + e * s), amp=True, cl=cl)
        fl += (p0 - pn).abs().sum(1)
        dz += (z0 - torch.logit(pn.clamp(1e-7, 1 - 1e-7))).abs().mean(1)
    return fl / NOISE_DRAWS, dz / NOISE_DRAWS


def _p_selection(p_init, it, n_iters):
    """Square-attack schedule of the window size (rescaled to n_iters)."""
    it = int(it / n_iters * 10000)
    for lim, div in ((10, 1), (50, 2), (200, 4), (500, 8), (1000, 16), (2000, 32),
                     (4000, 64), (6000, 128), (8000, 256)):
        if it <= lim:
            return p_init / div
    return p_init / 512


@torch.no_grad()
def square_probe(model, x, gen, cl=False):
    """Score-based L∞ Square attack on BCE w.r.t. the model's own clean prediction.
    Returns (losses at [0] + SQ_CHECK queries  (B, 1+len), final probabilities)."""
    B, C, H, W = x.shape
    dev = x.device
    e = v2.eps_t(SQ_EPS, x)
    p0 = probs_of(model, x, amp=True, cl=cl)
    y = (p0 > 0.5).float()
    out = [own_bce(p0, y)]
    init = (torch.randint(0, 2, (B, C, 1, W), device=dev, generator=gen).float() * 2 - 1) * e
    x_best = v2.clamp_valid(x + init)
    p_best = probs_of(model, x_best, amp=True, cl=cl)
    l_best = own_bce(p_best, y)
    rows = torch.arange(H, device=dev).view(1, H, 1)
    cols = torch.arange(W, device=dev).view(1, 1, W)
    record = {1: l_best.clone()}
    for q in range(2, SQ_QUERIES + 1):
        s = int(round(math.sqrt(_p_selection(SQ_PINIT, q - 1, SQ_QUERIES) * H * W)))
        s = min(max(s, 1), H - 1)
        vh = torch.randint(0, H - s + 1, (B,), device=dev, generator=gen).view(B, 1, 1)
        vw = torch.randint(0, W - s + 1, (B,), device=dev, generator=gen).view(B, 1, 1)
        mask = ((rows >= vh) & (rows < vh + s) & (cols >= vw) & (cols < vw + s)).unsqueeze(1).float()
        vals = (torch.randint(0, 2, (B, C, 1, 1), device=dev, generator=gen).float() * 2 - 1) * e
        x_new = v2.clamp_valid(x + (x_best - x) * (1 - mask) + vals * mask)
        p_new = probs_of(model, x_new, amp=True, cl=cl)
        l_new = own_bce(p_new, y)
        imp = l_new > l_best
        x_best = torch.where(imp.view(B, 1, 1, 1), x_new, x_best)
        p_best = torch.where(imp.view(B, 1), p_new, p_best)
        l_best = torch.maximum(l_new, l_best)
        if q in SQ_CHECK:
            record[q] = l_best.clone()
    out += [record[q] for q in SQ_CHECK]
    return torch.stack(out, 1), p_best, p0


def transfer_probes(model, x, direction, cl=False):
    """Probabilities of one target model under the ensemble-transfer probes x + ε·direction
    for every ε in PROBE_EPS → (E, B, 15)."""
    return torch.stack([probs_of(model, v2.clamp_valid(x + v2.eps_t(e, x) * direction), cl=cl)
                        for e in PROBE_EPS], 0)


# ─── GPU passes ───────────────────────────────────────────────────────────────

def _alloc(N, K, n_sq):
    C, E, L = v2.NUM_CLASSES, len(PROBE_EPS), len(SQ_CHECK) + 1
    return {"P_std": np.zeros((N, K, C), np.float32), "P0": np.zeros((N, K, C), np.float32),
            "PA": np.zeros((N, K, E, C), np.float32), "NF": np.zeros((N, K), np.float32),
            "ND": np.zeros((N, K), np.float32), "SQL": np.full((n_sq, K, L), np.nan, np.float32),
            "PSQ": np.full((n_sq, K, C), np.nan, np.float32),
            "SQP0": np.full((n_sq, K, C), np.nan, np.float32)}


def shadow_side(std_shadows, at_shadows, cache, img_idx, M, n_sq, device, seed, log):
    N, K = len(img_idx), len(at_shadows)
    o = _alloc(N, K, n_sq)
    crit = nn.BCEWithLogitsLoss()
    gen = v2.make_generator(device, seed + 11)
    pos, t0 = 0, time.time()
    for x in v2.loader(cache, img_idx, batch_size=BATCH, workers=4, device=device):
        x = x.to(device, non_blocking=True)
        b = slice(pos, pos + len(x))
        Mb = torch.from_numpy(M[b]).to(device)
        for k, f in enumerate(std_shadows):
            o["P_std"][b, k] = probs_of(f, x).cpu().numpy()
        G, P = [], []
        for k, f in enumerate(at_shadows):
            g, _, p = v2a._normalised_grad(f, x, crit)
            G.append(g)
            P.append(p)
            o["P0"][b, k] = p.cpu().numpy()
        G = torch.stack(G)
        n_in_sq = max(0, min(n_sq - pos, len(x)))
        for i, f in enumerate(at_shadows):
            w = (~Mb).float()
            w[:, i] = 0.0
            empty = w.sum(1) == 0
            if empty.any():
                w[empty] = 1.0
                w[empty, i] = 0.0
            d = torch.einsum("bk,kbchw->bchw", w, G).sign()
            o["PA"][b, i] = transfer_probes(f, x, d).permute(1, 0, 2).cpu().numpy()
            nf, nd = noise_probe(f, x, P[i], gen)
            o["NF"][b, i], o["ND"][b, i] = nf.cpu().numpy(), nd.cpu().numpy()
            if n_in_sq > 0:
                sql, psq, p0 = square_probe(f, x[:n_in_sq], gen)
                sb = slice(pos, pos + n_in_sq)
                o["SQL"][sb, i], o["PSQ"][sb, i] = sql.cpu().numpy(), psq.cpu().numpy()
                o["SQP0"][sb, i] = p0.cpu().numpy()
        del G
        pos += len(x)
        if pos % (BATCH * 25) < BATCH:
            log(f"    shadow side {pos}/{N} ({(time.time() - t0) / 60:.1f} min)")
    return o


def victim_side(victim, std_shadows, at_shadows, cache, img_idx, device, seed, log):
    N, K = len(img_idx), len(at_shadows)
    C, E, L = v2.NUM_CLASSES, len(PROBE_EPS), len(SQ_CHECK) + 1
    ref = _alloc(N, K, N)                        # reference models (all shadows) on eval images
    vic = {"P": np.zeros((N, C), np.float32), "PA": np.zeros((N, E, C), np.float32),
           "NF": np.zeros(N, np.float32), "ND": np.zeros(N, np.float32),
           "SQL": np.zeros((N, L), np.float32), "PSQ": np.zeros((N, C), np.float32),
           "SQP0": np.zeros((N, C), np.float32)}
    crit = nn.BCEWithLogitsLoss()
    gen = v2.make_generator(device, seed + 12)
    pos, t0 = 0, time.time()
    for x in v2.loader(cache, img_idx, batch_size=BATCH, workers=4, device=device):
        x = x.to(device, non_blocking=True)
        b = slice(pos, pos + len(x))
        pv = probs_of(victim, x, cl=True)
        vic["P"][b] = pv.cpu().numpy()
        for k, f in enumerate(std_shadows):
            ref["P_std"][b, k] = probs_of(f, x).cpu().numpy()
        G, P = [], []
        for k, f in enumerate(at_shadows):
            g, _, p = v2a._normalised_grad(f, x, crit)
            G.append(g)
            P.append(p)
            ref["P0"][b, k] = p.cpu().numpy()
        G = torch.stack(G)
        S = G.sum(0)
        vic["PA"][b] = transfer_probes(victim, x, S.sign(), cl=True).permute(1, 0, 2).cpu().numpy()
        nf, nd = noise_probe(victim, x, pv, gen, cl=True)
        vic["NF"][b], vic["ND"][b] = nf.cpu().numpy(), nd.cpu().numpy()
        sql, psq, p0 = square_probe(victim, x, gen, cl=True)
        vic["SQL"][b], vic["PSQ"][b], vic["SQP0"][b] = (sql.cpu().numpy(), psq.cpu().numpy(),
                                                        p0.cpu().numpy())
        for k, f in enumerate(at_shadows):       # reference: shadow k probed by the OTHER shadows
            ref["PA"][b, k] = transfer_probes(f, x, (S - G[k]).sign()).permute(1, 0, 2).cpu().numpy()
            nf, nd = noise_probe(f, x, P[k], gen)
            ref["NF"][b, k], ref["ND"][b, k] = nf.cpu().numpy(), nd.cpu().numpy()
            sql, psq, p0 = square_probe(f, x, gen)
            ref["SQL"][b, k], ref["PSQ"][b, k] = sql.cpu().numpy(), psq.cpu().numpy()
            ref["SQP0"][b, k] = p0.cpu().numpy()
        del G, S
        pos += len(x)
        if pos % (BATCH * 25) < BATCH:
            log(f"    victim side {pos}/{N} ({(time.time() - t0) / 60:.1f} min)")
    return vic, ref


# ─── Features (numpy) ─────────────────────────────────────────────────────────

def margins(P, yhat):
    """(2ŷ − 1)·logit(p) — positive = confident in the label ŷ."""
    return (2.0 * yhat - 1.0) * v2.logit_np(P)


def lira_features(m_t, m_refs, w):
    """m_t (n,C) target margins; m_refs (n,K,C) reference margins w.r.t. the SAME ŷ;
    w (n,K) reference weights (1 = model never trained on the image)."""
    wn = w / w.sum(1, keepdims=True)
    mu = np.einsum("nk,nkc->nc", wn, m_refs)
    var = np.einsum("nk,nkc->nc", wn, (m_refs - mu[:, None, :]) ** 2)
    z = (m_t - mu) / np.maximum(np.sqrt(var), SIGMA_FLOOR)
    agg = np.column_stack([z.mean(1), z.min(1), (m_t - mu).mean(1), m_t.mean(1), m_t.min(1)])
    return np.hstack([m_t, m_t - mu, z, agg]), z.mean(1)


def rob_features(P0, PA):
    pert, _, conf, _ = v2a.v2_afmia_features(P0, PA)
    return pert, conf


def query_features(SQL, PSQ, SQP0, NF, ND):
    inc = SQL[:, 1:] - SQL[:, :1]                         # loss increase after 1/10/30/60 queries
    d = np.abs(SQP0 - PSQ)
    dz = np.abs(v2.logit_np(SQP0) - v2.logit_np(PSQ))
    flips = ((PSQ > 0.5) != (SQP0 > 0.5)).sum(1)
    return np.column_stack([inc, SQL[:, :1], d.sum(1), dz.mean(1), flips, NF, ND])


def ref_weights(M, ra, rk):
    W = (~M[ra]).astype(np.float64)
    W[np.arange(len(ra)), rk] = 0.0
    empty = W.sum(1) == 0
    if empty.any():
        W[empty] = 1.0
        W[empty, rk[empty]] = 0.0
    return W


def ref_mean(F_all, ra, W):
    """F_all (N_A, K, d) → weighted mean over reference shadows for each row."""
    return np.einsum("nk,nkd->nd", W / W.sum(1, keepdims=True), F_all[ra])


def per_model(fn, *arrays):
    """Apply a (n, …) → (n, d) feature function to every model slice of (N, K, …) arrays."""
    K = arrays[0].shape[1]
    return np.stack([fn(*[a[:, k] for a in arrays]) for k in range(K)], 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Lab black-box MIAs on one victim")
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--attack_dir", required=True, help="results_v3/attacks/<run>")
    p.add_argument("--cache_dir", default=v2.DEFAULT_CACHE_DIR)
    p.add_argument("--n_square", type=int, default=N_Q, help="pool images with the Square probe")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out_dir = os.path.join(args.attack_dir, "lab")
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    global SQ_QUERIES, SQ_CHECK
    if args.smoke:
        SQ_QUERIES, SQ_CHECK = 6, [1, 3, 6]

    device = v2.get_device()
    torch.backends.cudnn.benchmark = True
    t_start = time.time()
    cache = mc.ImageCache(args.cache_dir)
    pool, eval_m, eval_nm = v2.attack_split(args.seed, smoke=args.smoke)
    pool_idx = cache.index_of(pool)
    eval_idx = np.concatenate([cache.index_of(eval_m), cache.index_of(eval_nm)])
    y_eval = np.concatenate([np.ones(len(eval_m)), np.zeros(len(eval_nm))]).astype(int)
    meta = v2.read_json(os.path.join(args.victim_dir, "model_meta.json"), {})

    plan_std, std_shadows = v2a.load_shadow_set(args.attack_dir, "std", device)
    plan_at, at_shadows = v2a.load_shadow_set(args.attack_dir, "at", device)
    if [s["train_indices"] for s in plan_std["shadows"]] != [s["train_indices"] for s in plan_at["shadows"]]:
        sys.exit("ERROR: the two shadow sets were trained on different pool samples")
    A, M = v2a.attack_subset(plan_at, args.seed, 200 if args.smoke else N_A)
    n_sq = min(len(A), 60 if args.smoke else args.n_square)
    rows = v2a.balanced_rows(M, args.seed)
    ra_, rk, ry = rows[:, 0], rows[:, 1], rows[:, 2]
    victim = v2.load_victim(os.path.join(args.victim_dir, "model.pth"), device)
    for prm in victim.parameters():
        prm.requires_grad_(False)
    log(f"[lab] {args.victim_dir}: {len(rows)} rows ({n_sq} images with the Square probe), "
        f"eval {len(eval_m)} + {len(eval_nm)}; Square ε={SQ_EPS:g}/255, {SQ_QUERIES} queries")

    sp, vp = os.path.join(cache_dir, "shadow_side.npz"), os.path.join(cache_dir, "victim_side.npz")
    if os.path.exists(sp):
        sh = dict(np.load(sp))
        log("  shadow side loaded from cache")
    else:
        log("  shadow side …")
        sh = shadow_side(std_shadows, at_shadows, cache, pool_idx[A], M, n_sq, device, args.seed, log)
        v2.save_npz(sp, A=A, M=M, **sh)
    if os.path.exists(vp):
        z = np.load(vp)
        vic = {k[2:]: z[k] for k in z.files if k.startswith("v_")}
        ref = {k[2:]: z[k] for k in z.files if k.startswith("r_")}
        log("  victim side loaded from cache")
    else:
        log("  victim side …")
        vic, ref = victim_side(victim, std_shadows, at_shadows, cache, eval_idx, device, args.seed, log)
        v2.save_npz(vp, **{f"v_{k}": v for k, v in vic.items()}, **{f"r_{k}": v for k, v in ref.items()})

    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    K = M.shape[1]
    W = ref_weights(M, ra_, rk)

    # ── LiRA-cal confidence (standard shadows) ────────────────────────────────
    P_std = f64(sh["P_std"])
    yhat_r = (P_std[ra_, rk] > 0.5).astype(np.float64)
    m_t = margins(P_std[ra_, rk], yhat_r)
    m_refs = margins(P_std[ra_], yhat_r[:, None, :])
    X_lira_tr, s_lira_tr = lira_features(m_t, m_refs, W)
    Pv = f64(vic["P"])
    yhat_v = (Pv > 0.5).astype(np.float64)
    X_lira_ev, s_lira_ev = lira_features(margins(Pv, yhat_v), margins(f64(ref["P_std"]), yhat_v[:, None, :]),
                                         np.ones((len(Pv), K)))

    # ── Robustness-cal (AT shadows) ───────────────────────────────────────────
    F_pert = per_model(lambda a, b: rob_features(a, b)[0], f64(sh["P0"]), f64(sh["PA"]))
    F_conf = per_model(lambda a, b: rob_features(a, b)[1], f64(sh["P0"]), f64(sh["PA"]))
    pert_t, conf_t = F_pert[ra_, rk], F_conf[ra_, rk]
    pert_cal_t, conf_cal_t = pert_t - ref_mean(F_pert, ra_, W), conf_t - ref_mean(F_conf, ra_, W)
    pert_v, conf_v = rob_features(Pv, f64(vic["PA"]))
    Wv = np.ones((len(Pv), K))
    R_pert = per_model(lambda a, b: rob_features(a, b)[0], f64(ref["P0"]), f64(ref["PA"]))
    R_conf = per_model(lambda a, b: rob_features(a, b)[1], f64(ref["P0"]), f64(ref["PA"]))
    pert_cal_v = pert_v - np.einsum("nk,nkd->nd", Wv / K, R_pert)
    conf_cal_v = conf_v - np.einsum("nk,nkd->nd", Wv / K, R_conf)

    # ── Query robustness (Square + noise) ─────────────────────────────────────
    sq_rows = ra_ < n_sq
    F_q = per_model(query_features, f64(sh["SQL"]), f64(sh["PSQ"]), f64(sh["SQP0"]),
                    f64(sh["NF"][:n_sq]), f64(sh["ND"][:n_sq]))
    Wq = W[sq_rows]
    q_t = F_q[ra_[sq_rows], rk[sq_rows]]
    q_cal_t = q_t - np.einsum("nk,nkd->nd", Wq / Wq.sum(1, keepdims=True), F_q[ra_[sq_rows]])
    q_v = query_features(f64(vic["SQL"]), f64(vic["PSQ"]), f64(vic["SQP0"]), f64(vic["NF"]),
                         f64(vic["ND"]))
    R_q = per_model(query_features, f64(ref["SQL"]), f64(ref["PSQ"]), f64(ref["SQP0"]),
                    f64(ref["NF"]), f64(ref["ND"]))
    q_cal_v = q_v - R_q.mean(1)

    sets = [
        (FS_LIRA, X_lira_tr, ry, X_lira_ev),
        (FS_ROB, np.hstack([pert_t, pert_cal_t]), ry, np.hstack([pert_v, pert_cal_v])),
        (FS_ROB_CONF, np.hstack([pert_t, pert_cal_t, conf_t, conf_cal_t]), ry,
         np.hstack([pert_v, pert_cal_v, conf_v, conf_cal_v])),
        (FS_QUERY, np.hstack([q_t, q_cal_t]), ry[sq_rows], np.hstack([q_v, q_cal_v])),
        (FS_ALL, np.hstack([X_lira_tr[sq_rows], pert_t[sq_rows], pert_cal_t[sq_rows],
                            conf_t[sq_rows], conf_cal_t[sq_rows], q_t, q_cal_t]), ry[sq_rows],
         np.hstack([X_lira_ev, pert_v, pert_cal_v, conf_v, conf_cal_v, q_v, q_cal_v])),
    ]
    results, all_scores = [], {}
    for name, X_tr, y_tr, X_ev in sets:
        if len(np.unique(y_tr)) < 2:
            log(f"  skip {name}: one class only in the training rows")
            continue
        r, s = v2a.evaluate_feature_set(name, X_tr, y_tr, X_ev, y_eval, args.seed, log)
        results += r
        all_scores.update(s)

    # ── Classifier-free scores (direction fixed: confident / stable = member) ─
    ref_e4 = PROBE_EPS.index(4.0) * 5          # fl_sum_e4 column in the perturbation block
    thr = {
        "lira_mean_z": (s_lira_tr, s_lira_ev, None),
        "neg_cal_fl_sum_e4": (-pert_cal_t[:, ref_e4], -pert_cal_v[:, ref_e4], None),
        "neg_cal_dz_mean_e4": (-pert_cal_t[:, ref_e4 + 2], -pert_cal_v[:, ref_e4 + 2], None),
        "neg_sq_loss_inc_final": (-q_t[:, len(SQ_CHECK) - 1], -q_v[:, len(SQ_CHECK) - 1], sq_rows),
        "neg_cal_sq_loss_inc_final": (-q_cal_t[:, len(SQ_CHECK) - 1], -q_cal_v[:, len(SQ_CHECK) - 1],
                                      sq_rows),
        "neg_noise_fl": (-q_t[:, -2], -q_v[:, -2], sq_rows),
        "neg_cal_noise_fl": (-q_cal_t[:, -2], -q_cal_v[:, -2], sq_rows),
    }
    thr_rows = []
    for k, (s_tr, s_ev, mask) in thr.items():
        y_tr = ry if mask is None else ry[mask]
        m = v2.score_metrics(y_eval, s_ev)
        sh_auc = v2.score_metrics(y_tr, s_tr)["auc"] if len(np.unique(y_tr)) == 2 else float("nan")
        thr_rows.append({"name": f"threshold: {k}", "key": k, **m, "shadow_row_auc": sh_auc})
        all_scores[f"threshold|{k}"] = np.asarray(s_ev, dtype=np.float32)

    out = {"attack": "lab", "victim_dir": args.victim_dir, "seed": args.seed,
           "config": {"rows": int(len(rows)), "square_rows": int(sq_rows.sum()), "n_square_images": n_sq,
                      "square_eps255": SQ_EPS, "square_queries": SQ_QUERIES, "square_checkpoints": SQ_CHECK,
                      "noise_eps255": NOISE_EPS, "noise_draws": NOISE_DRAWS, "probe_eps255": PROBE_EPS,
                      "eval_members": int(len(eval_m)), "eval_nonmembers": int(len(eval_nm))},
           "results": results, "threshold_scores": thr_rows,
           "diagnostics": {
               "victim_square_loss_increase_final": {
                   "members": float(q_v[y_eval == 1, len(SQ_CHECK) - 1].mean()),
                   "nonmembers": float(q_v[y_eval == 0, len(SQ_CHECK) - 1].mean())},
               "victim_noise_fl": {"members": float(q_v[y_eval == 1, -2].mean()),
                                   "nonmembers": float(q_v[y_eval == 0, -2].mean())}},
           "runtime_s": round(time.time() - t_start, 1),
           "victim_meta": {k: v for k, v in meta.items() if k != "label_names"}}
    v2.save_npz(os.path.join(out_dir, "scores.npz"), y_true=y_eval.astype(np.int8), **all_scores)
    v2.write_json(os.path.join(out_dir, "results.json"), out)
    v2a.write_txt(os.path.join(out_dir, "results.txt"), f"lab on {args.victim_dir} (seed {args.seed})",
                  results, thr_rows)
    log(f"[DONE] lab → {out_dir} ({out['runtime_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
