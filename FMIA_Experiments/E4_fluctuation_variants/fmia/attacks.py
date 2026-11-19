"""Membership inference attacks on cached signals (numpy / scikit-learn only).

One call = one (setting, target model): the target's signals, the reference
models' signals with their membership masks (the attacker trained them, so it
knows them), and which probes the access level allows.  Every score is
"higher = more likely a member".  Anything an attack tunes (feature weights,
probe choice, sign, RMIA's a) is tuned on the REFERENCE models only — each
reference in turn plays a pseudo-target against the others — never on the
target's labels.

Existing attacks (baselines)
  loss         Yeom et al. 2018 — loss threshold
  mentr        Song & Mittal 2021 — modified-entropy threshold
  shokri       Shokri et al. 2017 — shadow models + classifier on the clean probability
               vector, max, argmax (the paper's "Original MIA")
  lira_off/on  Carlini et al. 2022 — per-example Gaussian likelihood ratio on the
               logit-scaled confidence φ (clean + flip queries), fixed variance;
               lira_on_pv: per-example variance; lira_aug_on: 18 augmented queries
  rmia_off/on  Zarifzadeh et al. 2024 — Pr(x|θ)/Pr(x) with Pr(x) from the references
               (offline: ½((1+a)·mean_OUT + (1−a)), a tuned on references).  With γ = 1
               the RMIA score is a monotone function of this ratio, so the ranking is
               identical to the pairwise test against population samples z.
  song_adv     Song et al. 2019 — loss threshold on an adversarial example (the
               robustness-based attack designed for adversarially trained models)
  labelonly    Choquette-Choo et al. 2021 — fraction of correct predictions over
               augmented / noisy queries (label-only)
  gradnorm     white-box input-gradient norm

Fluctuation attacks (ours)
  fmia_paper   the paper's Fluctuation MIA: GB on [|p_clean − p_probe| per class, max, argmax]
  fl_raw       threshold on one fluctuation (probe + sign chosen on references)
  fl_cal_off/on  per-example-calibrated fluctuation of one probe (z-score vs OUT
               references / Gaussian likelihood ratio IN vs OUT)
  fl_gb        GB on all raw fluctuation features (uncalibrated)
  cfmia_*      CF-MIA, Calibrated multi-probe Fluctuation MIA (the proposed attack):
               every fluctuation statistic of every allowed probe is calibrated per
               example against the references (offline z-score or online LLR), and a
               logistic regression learned on pseudo-target references combines them.
                 cfmia_fl_off/on   fluctuation statistics only
                 cfmia_off/on      + the confidence statistics LiRA uses
                 cfmia_q_on        query-only probes (noise/augmentation, no surrogate
                                   gradients) + confidence
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auc, summarize

LABELS = {
    "loss": "Loss threshold (Yeom+ 2018)",
    "mentr": "Modified entropy (Song & Mittal 2021)",
    "shokri": "Shadow-model MIA = paper 'Original MIA' (Shokri+ 2017)",
    "lira_off": "LiRA offline (Carlini+ 2022)",
    "lira_on": "LiRA online (Carlini+ 2022)",
    "lira_on_pv": "LiRA online, per-example variance",
    "lira_aug_on": "LiRA online, 18 augmented queries",
    "rmia_off": "RMIA offline (Zarifzadeh+ 2024)",
    "rmia_on": "RMIA online (Zarifzadeh+ 2024)",
    "song_adv": "Adversarial-loss threshold (Song+ 2019)",
    "labelonly": "Label-only aug./noise (Choquette-Choo+ 2021)",
    "gradnorm": "Input-gradient norm (white-box)",
    "fmia_paper": "Fluctuation MIA, paper version (GB on |Δp|)",
    "fl_raw": "Fluctuation threshold (best probe)",
    "fl_cal_off": "Calibrated fluctuation, offline (best probe)",
    "fl_cal_on": "Calibrated fluctuation, online (best probe)",
    "fl_gb": "Fluctuation features → GB (uncalibrated)",
    "cfmia_fl_off": "CF-MIA offline, fluctuation only",
    "cfmia_fl_on": "CF-MIA online, fluctuation only",
    "cfmia_q_on": "CF-MIA online, query-only probes",
    "cfmia_off": "CF-MIA offline (proposed)",
    "cfmia_on": "CF-MIA online (proposed)",
}
EXISTING = ["loss", "mentr", "shokri", "lira_off", "lira_on", "lira_on_pv", "lira_aug_on",
            "rmia_off", "rmia_on", "song_adv", "labelonly", "gradnorm"]
OURS = ["fmia_paper", "fl_raw", "fl_cal_off", "fl_cal_on", "fl_gb", "cfmia_fl_off", "cfmia_fl_on",
        "cfmia_q_on", "cfmia_off", "cfmia_on"]
OFFLINE = {"loss", "mentr", "shokri", "lira_off", "rmia_off", "song_adv", "labelonly", "gradnorm",
           "fmia_paper", "fl_raw", "fl_cal_off", "fl_gb", "cfmia_fl_off", "cfmia_off"}
MAIN_ROC = ["loss", "shokri", "lira_on", "lira_off", "rmia_on", "rmia_off", "song_adv",
            "fmia_paper", "cfmia_on", "cfmia_off", "cfmia_fl_on"]


# ─── signal access ────────────────────────────────────────────────────────────

class Sig:
    """Lazy view of one model's signal file."""

    def __init__(self, path, keep, name):
        self.path, self.name = path, name
        self.keep = np.asarray(keep, bool)
        self._z = np.load(path)
        self.keys = set(self._z.files)
        self._c = {}

    def __getitem__(self, k):
        if k not in self._c:
            self._c[k] = np.asarray(self._z[k], dtype=np.float32)
        return self._c[k]

    def has(self, k):
        return k in self.keys


def phi_avg(s):
    return 0.5 * (s["clean.phi"] + s["hflip.phi"]) if s.has("hflip.phi") else s["clean.phi"]


def ptrue_avg(s):
    return 0.5 * (s["clean.ptrue"] + s["hflip.ptrue"]) if s.has("hflip.ptrue") else s["clean.ptrue"]


def clean_probs(s, multilabel):
    z = s["clean.logits"].astype(np.float64)
    if multilabel:
        return 1 / (1 + np.exp(-z))
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ─── probes allowed per access level ──────────────────────────────────────────

QUERY_PREFIX = ("gauss:", "uniform:")


def is_query_probe(name):
    return name in ("hflip", "augset") or name.startswith(QUERY_PREFIX)


def allowed_probes(probe_names, sources, white_box):
    out = []
    for n in probe_names:
        if is_query_probe(n):
            out.append(n)
        elif n.startswith("self"):
            if white_box:
                out.append(n)
        elif n.startswith(("tr:", "tr_own:")):
            if n.split(":")[1] in sources:
                out.append(n)
    return out


def fluct_columns(probes):
    """(column key, kind) for every fluctuation statistic of the perturbation probes."""
    cols = []
    for p in probes:
        if p in ("hflip", "augset"):
            continue
        cols.append((p, "drop"))       # φ_clean − φ_probe
        cols.append((p, "l1"))         # Σ|p_clean − p_probe| (paper)
        if p.startswith(QUERY_PREFIX):
            cols.append((p, "phi_std"))
    return cols


def column(s, col):
    p, kind = col
    if kind == "drop":
        return s["clean.phi"] - s[f"{p}.phi"]
    if kind == "conf":
        return phi_avg(s) if p == "clean" else s[f"{p}.phi"]
    return s[f"{p}.{kind}"]


def matrix(s, cols):
    return np.stack([column(s, c) for c in cols], 1).astype(np.float32)


# ─── per-example calibration against reference models ─────────────────────────

def _masked_stats(V, m):
    """V (M, N, D), m (M, N) bool → mean, var, count per (N, D); NaN where undefined."""
    w = m[..., None].astype(np.float32)
    n = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (V * w).sum(0) / n
        var = (((V - mu[None]) ** 2) * w).sum(0) / (n - 1)
    mu = np.where(n == 0, np.nan, mu)
    var = np.where(n < 2, np.nan, var)
    return mu, var, n


def _fill_mean(mu, other_mu, delta):
    bad = ~np.isfinite(mu)
    if bad.any():
        fill = other_mu + delta[None, :]
        glob = np.nanmean(np.where(np.isfinite(other_mu), other_mu, np.nan), 0)
        fill = np.where(np.isfinite(fill), fill, glob[None, :])
        mu = np.where(bad, fill, mu)
    return mu


def calibrate(Vt, Vr, Kr, mode, fixed_var=True, min_sd=1e-4):
    """Offline: z = (v − μ_OUT(x)) / σ_OUT.  Online: log N(v; μ_IN, σ_IN) − log N(v; μ_OUT, σ_OUT).
    Vt (N, D) target values; Vr (M, N, D) reference values; Kr (M, N) reference membership."""
    Vt = Vt.astype(np.float32)
    mu_o, var_o, _ = _masked_stats(Vr, ~Kr)
    if mode == "online":
        mu_i, var_i, _ = _masked_stats(Vr, Kr)
        with np.errstate(invalid="ignore"):
            delta = np.nanmean(mu_i - mu_o, 0)
        delta = np.nan_to_num(delta)
        mu_i = _fill_mean(mu_i, mu_o, delta)
        mu_o = _fill_mean(mu_o, mu_i, -delta)
    else:
        mu_o = _fill_mean(mu_o, mu_o, np.zeros(Vt.shape[1], np.float32))
    gvar_o = np.nan_to_num(np.nanmean(var_o, 0), nan=1.0)
    if fixed_var:
        sd_o = np.sqrt(np.maximum(gvar_o, min_sd ** 2))[None, :]
    else:
        sd_o = np.sqrt(np.maximum(np.where(np.isfinite(var_o), var_o, gvar_o[None, :]), min_sd ** 2))
    if mode == "offline":
        return (Vt - mu_o) / sd_o
    gvar_i = np.nan_to_num(np.nanmean(var_i, 0), nan=1.0)
    if fixed_var:
        sd_i = np.sqrt(np.maximum(gvar_i, min_sd ** 2))[None, :]
    else:
        sd_i = np.sqrt(np.maximum(np.where(np.isfinite(var_i), var_i, gvar_i[None, :]), min_sd ** 2))
    return (-0.5 * ((Vt - mu_i) / sd_i) ** 2 - np.log(sd_i)) - (-0.5 * ((Vt - mu_o) / sd_o) ** 2 - np.log(sd_o))


def spread(M, n):
    """n reference indices spread evenly over 0..M-1 (references are listed family by family,
    so taking the first n would draw them all from one family when families are pooled)."""
    if n >= M:
        return list(range(M))
    return sorted(set(np.linspace(0, M - 1, n).round().astype(int).tolist()))


def _loo(Vr, Kr, j):
    others = [i for i in range(len(Vr)) if i != j]
    return Vr[j], Vr[others], Kr[others]


# ─── learned components ───────────────────────────────────────────────────────

def _rows(rng, n, k):
    return rng.choice(n, min(n, k), replace=False)


def gb_attack(feat_fn, target, refs, seed, rows_per_ref=10000, max_refs=8):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in spread(len(refs), max_refs):
        r = refs[i]
        F = feat_fn(r)
        idx = _rows(rng, len(F), rows_per_ref)
        X.append(F[idx])
        Y.append(r.keep[idx])
    X = np.nan_to_num(np.concatenate(X), nan=0.0, posinf=1e6, neginf=-1e6)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    clf.fit(X, np.concatenate(Y))
    Ft = np.nan_to_num(feat_fn(target), nan=0.0, posinf=1e6, neginf=-1e6)
    return clf.predict_proba(Ft)[:, 1]


def stacked(Vt, Vr, Kr, mode, seed, n_pseudo=4, rows=20000, C=1.0):
    """Calibrate every column, then learn a linear combination on pseudo-target references."""
    M = len(Vr)
    if M < 3 or (mode == "online" and M < 4):
        return None
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for j in spread(M, n_pseudo):
        vj, vr, kr = _loo(Vr, Kr, j)
        Fj = calibrate(vj, vr, kr, mode)
        idx = _rows(rng, len(Fj), rows)
        X.append(Fj[idx])
        Y.append(Kr[j][idx])
    X = np.clip(np.nan_to_num(np.concatenate(X)), -1e4, 1e4)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(X, np.concatenate(Y))
    Ft = np.clip(np.nan_to_num(calibrate(Vt, Vr, Kr, mode)), -1e4, 1e4)
    return clf.decision_function(Ft)


def pseudo_auc_per_column(Vr, Kr, mode, n_pseudo=4):
    """Mean pseudo-target AUC of each calibrated (or raw, mode=None) column on references."""
    M, _, D = Vr.shape
    idx = spread(M, n_pseudo)
    aucs = np.zeros((len(idx), D))
    for a, j in enumerate(idx):
        vj, vr, kr = _loo(Vr, Kr, j)
        F = vj if mode is None else calibrate(vj, vr, kr, mode)
        for d in range(D):
            aucs[a, d] = auc(Kr[j], F[:, d])
    return np.nanmean(aucs, 0)


def rmia_scores(Pt, Pr, Kr, a=None):
    mu_o, _, _ = _masked_stats(Pr[..., None], ~Kr)
    mu_o = mu_o[:, 0]
    g_out = np.nanmean(mu_o)
    mu_o = np.where(np.isfinite(mu_o), mu_o, g_out)
    if a is None:                                           # online
        mu_i, _, _ = _masked_stats(Pr[..., None], Kr)
        mu_i = mu_i[:, 0]
        mu_i = np.where(np.isfinite(mu_i), mu_i, np.nanmean(mu_i))
        px = 0.5 * (mu_i + mu_o)
    else:
        px = 0.5 * ((1 + a) * mu_o + (1 - a))
    return Pt / np.maximum(px, 1e-12)


# ─── the attack suite ─────────────────────────────────────────────────────────

def run_attacks(target, refs, probes, white_box, multilabel, seed=0, song_probe=None,
                paper_probe=None, detail=True, n_pseudo=4):
    """Returns (results list, info dict).  target: Sig; refs: list[Sig]; probes: allowed names."""
    y = target.keep
    M = len(refs)
    Kr = np.stack([r.keep for r in refs]) if M else None
    res, info = [], {"n_refs": M, "probes": probes}

    def add(name, score, category=None, label=None, **extra):
        if score is None:
            return
        m = summarize(y, score, with_roc=name in MAIN_ROC)
        cat = category or ("existing" if name in EXISTING else "ours")
        m.update(attack=name, label=label or LABELS.get(name, name), category=cat,
                 offline=name in OFFLINE, **extra)
        res.append(m)

    # ---- reference-free baselines
    add("loss", -target["clean.loss"])
    add("mentr", -target["clean.mentr"])
    q_corr = [p for p in probes if p == "augset" or p.startswith(QUERY_PREFIX)]
    if q_corr:
        add("labelonly", np.mean([target[f"{p}.corr"] for p in q_corr], 0))
    if song_probe:
        add("song_adv", -target[f"{song_probe}.loss"], probe=song_probe)

    if M == 0:
        return res, info

    # ---- reference-based baselines
    add("shokri", gb_attack(lambda s: _shokri_feats(s, multilabel), target, refs, seed))
    phi_t = phi_avg(target)[:, None]
    phi_r = np.stack([phi_avg(r) for r in refs])[..., None]
    if M >= 2:
        add("lira_off", calibrate(phi_t, phi_r, Kr, "offline")[:, 0])
        add("lira_on", calibrate(phi_t, phi_r, Kr, "online")[:, 0])
        add("lira_on_pv", calibrate(phi_t, phi_r, Kr, "online", fixed_var=False)[:, 0])
        if target.has("augset.phi"):
            add("lira_aug_on", calibrate(target["augset.phi"][:, None],
                                         np.stack([r["augset.phi"] for r in refs])[..., None], Kr,
                                         "online")[:, 0])
        Pt = ptrue_avg(target)
        Pr = np.stack([ptrue_avg(r) for r in refs])
        add("rmia_on", rmia_scores(Pt, Pr, Kr))
        best_a, best = 0.5, -1.0
        for a in np.linspace(0, 1, 11):            # tune a on pseudo-target references
            vals = []
            for j in spread(M, n_pseudo):
                p_j, p_rest, k_rest = _loo(Pr, Kr, j)
                vals.append(auc(Kr[j], rmia_scores(p_j, p_rest, k_rest, a=a)))
            s = float(np.nanmean(vals))
            if s > best:
                best, best_a = s, float(a)
        info["rmia_a"] = float(best_a)
        add("rmia_off", rmia_scores(Pt, Pr, Kr, a=best_a), a=float(best_a))
    if white_box and target.has("self.gradnorm"):
        gr = np.stack([r["self.gradnorm"] for r in refs])
        sgn = 1.0 if pseudo_auc_per_column(-gr[..., None], Kr, None, n_pseudo)[0] >= 0.5 else -1.0
        add("gradnorm", -sgn * target["self.gradnorm"])

    # ---- fluctuation attacks
    fl_cols = fluct_columns(probes)
    fl_cols = [c for c in fl_cols if target.has(f"{c[0]}.{'phi' if c[1] == 'drop' else c[1]}")]
    conf_cols = [("clean", "conf")] + ([("augset", "conf")] if "augset" in probes else [])
    if paper_probe and target.has(f"{paper_probe}.dp"):
        add("fmia_paper", gb_attack(lambda s: _paper_feats(s, paper_probe, multilabel), target, refs, seed),
            probe=paper_probe)
    if not fl_cols:
        return res, info

    Vt = matrix(target, fl_cols)
    Vr = np.stack([matrix(r, fl_cols) for r in refs])
    names = [f"{p}|{k}" for p, k in fl_cols]
    # single-column attacks: probe + sign chosen on the references
    a_raw = pseudo_auc_per_column(-Vr, Kr, None, n_pseudo) if M >= 2 else None
    if a_raw is not None:
        sgn = np.where(a_raw >= 0.5, 1.0, -1.0)
        j = int(np.nanargmax(np.abs(a_raw - 0.5)))
        add("fl_raw", -sgn[j] * Vt[:, j], column=names[j])
        if M >= 3:
            a_off = pseudo_auc_per_column(Vr, Kr, "offline", n_pseudo)
            s_off = np.where(a_off >= 0.5, 1.0, -1.0)
            j = int(np.nanargmax(np.abs(a_off - 0.5)))
            Z = calibrate(Vt, Vr, Kr, "offline")
            add("fl_cal_off", s_off[j] * Z[:, j], column=names[j])
            L = calibrate(Vt, Vr, Kr, "online")
            a_on = pseudo_auc_per_column(Vr, Kr, "online", n_pseudo)
            j_on = int(np.nanargmax(a_on))
            add("fl_cal_on", L[:, j_on], column=names[j_on])
            if detail:
                for d, nm in enumerate(names):
                    add(f"raw:{nm}", -sgn[d] * Vt[:, d], category="detail", label=f"raw {nm}")
                    add(f"cal_off:{nm}", s_off[d] * Z[:, d], category="detail", label=f"cal-off {nm}")
                    add(f"cal_on:{nm}", L[:, d], category="detail", label=f"cal-on {nm}")
    gb_cols = conf_cols + fl_cols
    add("fl_gb", gb_attack(lambda s: matrix(s, gb_cols), target, refs, seed))

    # CF-MIA
    add("cfmia_fl_off", stacked(Vt, Vr, Kr, "offline", seed, n_pseudo))
    add("cfmia_fl_on", stacked(Vt, Vr, Kr, "online", seed, n_pseudo))
    Ct = matrix(target, conf_cols)
    Cr = np.stack([matrix(r, conf_cols) for r in refs])
    VtA, VrA = np.concatenate([Ct, Vt], 1), np.concatenate([Cr, Vr], 2)
    add("cfmia_off", stacked(VtA, VrA, Kr, "offline", seed, n_pseudo))
    add("cfmia_on", stacked(VtA, VrA, Kr, "online", seed, n_pseudo))
    q_idx = [i for i, c in enumerate(fl_cols) if is_query_probe(c[0])]
    if q_idx and len(q_idx) < len(fl_cols):
        sel = list(range(len(conf_cols))) + [len(conf_cols) + i for i in q_idx]
        add("cfmia_q_on", stacked(VtA[:, sel], VrA[:, :, sel], Kr, "online", seed, n_pseudo))
    elif q_idx:
        info["cfmia_q_on"] = "same as cfmia_on (only query probes allowed)"
    return res, info


def _shokri_feats(s, multilabel):
    P = clean_probs(s, multilabel)
    return np.concatenate([P, P.max(1, keepdims=True), P.argmax(1)[:, None].astype(np.float64)], 1)


def _paper_feats(s, probe, multilabel):
    P = clean_probs(s, multilabel)
    return np.concatenate([s[f"{probe}.dp"], P.max(1, keepdims=True),
                           P.argmax(1)[:, None].astype(np.float64)], 1)
