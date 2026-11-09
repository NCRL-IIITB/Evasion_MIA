"""
aggregate_results.py
====================
Collect every v2 result into one report.  Works on a single account's output
(one victim) and on the merged output of both accounts (C and D), where it
adds the paired comparison.

Reads   <results>/victims/*/model_meta.json, history.csv
        <results>/eval/*/{evasion.json, threshold_mia.json, threshold_scores.npz}
        <results>/attacks/*/{shadow_mia_std, shadow_mia_at, afmia}/{results.json, scores.npz}
Writes  <results>/summary/
          REPORT.md                 everything below in one file
          per_run.csv               one row per victim, headline metrics
          attack_results.csv        every feature set × classifier × victim
          paired.csv                D − C differences with paired-bootstrap 95 % CIs
          attack_comparison_*.png   the three attacks, C vs D (one figure per D variant)
          gap_matched.png           threshold-MIA AUC vs clean loss gap over checkpoints

Paired bootstrap: C and D trained with the same seed are attacked on the SAME
5k + 5k evaluation images, so members and non-members are resampled with the
same indices for both victims; the CI reflects evaluation-sample noise for
these two trained models (training-seed noise needs more seeds).
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as v2  # noqa: E402

PRIMARY = "Gradient Boosting"
ATTACK_DIRS = ["shadow_mia_std", "afmia", "shadow_mia_at"]
HEADLINE_MIA = [
    ("Shadow-MIA", "Shadow-MIA (paper features, standard shadows)"),
    ("Shadow-MIA + BCVA", "Shadow-MIA + BCVA (paper: Original MIA + Variance)"),
    ("AFMIA (paper features) + BCVA", "AFMIA paper features + BCVA (paper: Fluctuation + Variance)"),
    ("AFMIA-v2 (perturbation only)", "AFMIA-v2, perturbation response only"),
    ("AFMIA-v2 (+ clean confidence)", "AFMIA-v2 + clean confidence"),
    ("AFMIA (paper features)", "AFMIA, paper features"),
    ("Shadow-MIA (AT shadows)", "Shadow-MIA, AT shadows (defence-aware)"),
]
THRESH_LABELS = {"loss": "Loss (true labels)", "certainty": "Confidence (label-free)",
                 "adv_loss": "Adversarial loss, FGSM", "adv_loss_pgd": "Adversarial loss, PGD-5",
                 "fluct_e1": "Fluctuation ε=1/255", "fluct_e4": "Fluctuation ε=4/255",
                 "dlogit_e4": "Logit fluctuation ε=4/255"}
HEAD_EPS = "4"
# The paper's four feature strategies (Tables II / III) and where v2 computes them
PAPER_STRATEGIES = [("Original MIA", "Shadow-MIA"),
                    ("Original MIA + Variance", "Shadow-MIA + BCVA"),
                    ("Fluctuation", "AFMIA (paper features)"),
                    ("Fluctuation + Variance", "AFMIA (paper features) + BCVA")]
CLASSIFIERS = ["Gradient Boosting", "Random Forest", "MLP", "Logistic Regression"]
PAPER_EPS = ["0.001", "0.002", "0.005", "0.01", "0.02"]

C_STD, C_AT = "#2a78d6", "#eb6834"          # categorical slots 1–2 (validated, light surface)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ─── Collect ──────────────────────────────────────────────────────────────────

def _training(meta):
    return "fgsm" if meta.get("training_type") == "adversarial" else "standard"


def collect(R):
    runs = {}
    for meta_path in sorted(glob.glob(os.path.join(R, "victims", "*", "model_meta.json"))):
        run = os.path.basename(os.path.dirname(meta_path))
        try:
            cell_id, seed = v2.parse_run(run)
        except ValueError:
            continue
        meta = v2.read_json(meta_path, {})
        r = {"run": run, "cell_id": cell_id, "seed": seed, "training": _training(meta),
             "meta": meta, "missing": []}
        hist = os.path.join(R, "victims", run, "history.csv")
        r["history"] = pd.read_csv(hist) if os.path.exists(hist) else None
        ev = os.path.join(R, "eval", run)
        r["evasion"] = v2.read_json(os.path.join(ev, "evasion.json"))
        r["threshold"] = v2.read_json(os.path.join(ev, "threshold_mia.json"))
        tsp = os.path.join(ev, "threshold_scores.npz")
        r["threshold_scores"] = dict(np.load(tsp)) if os.path.exists(tsp) else None
        if r["evasion"] is None:
            r["missing"].append("eval/evasion.json")
        if r["threshold"] is None:
            r["missing"].append("eval/threshold_mia.json")
        r["attacks"], r["scores"] = {}, {}
        for a in ATTACK_DIRS:
            d = os.path.join(R, "attacks", run, a)
            res = v2.read_json(os.path.join(d, "results.json"))
            if res is None:
                r["missing"].append(f"attacks/{a}/results.json")
                continue
            r["attacks"][a] = res
            sp = os.path.join(d, "scores.npz")
            if os.path.exists(sp):
                z = np.load(sp)
                r["scores"].update({k: z[k] for k in z.files if k != "y_true"})
                r["scores"]["__y__"] = z["y_true"]
        runs[run] = r
    return runs


def attack_rows(r):
    out = []
    for a, res in r["attacks"].items():
        for x in res["results"]:
            out.append({"run": r["run"], "attack_dir": a, **{k: x[k] for k in (
                "feature_set", "classifier", "auc", "tpr_at_1pct_fpr", "tpr_at_0.1pct_fpr",
                "accuracy", "acc_median", "best_threshold_acc", "precision", "recall", "f1")}})
    return out


def find_result(r, feature_set, classifier=PRIMARY):
    for res in r["attacks"].values():
        for x in res["results"]:
            if x["feature_set"] == feature_set and x["classifier"] == classifier:
                return x
    return None


def run_row(r):
    m, h = r["meta"], r["history"]
    row = {"run": r["run"], "cell_id": r["cell_id"], "seed": r["seed"], "training": r["training"],
           "method": m.get("adversarial_method"), "train_eps255": m.get("train_eps255"),
           "alpha": m.get("adversarial_alpha"), "train_acc": m.get("final_train_acc"),
           "val_auc": m.get("final_val_auc"), "fgsm_val_auc": m.get("final_adv_val_auc"),
           "pgd_val_auc": m.get("final_pgd_val_auc"), "memorization_gap": m.get("memorization_gap"),
           "imported_from": m.get("imported_from")}
    if h is not None and "adv_val_auc" in h and len(h) > 1:
        row["max_epoch_swing_fgsm_val_auc"] = float(h["adv_val_auc"].diff().abs().max())
    ev = r["evasion"]
    if ev:
        row["test_auc_clean"] = ev["clean"]["auc_macro"]
        for e, v in ev.get("fgsm", {}).items():
            row[f"fgsm_flip_e{e}"] = v["flip_rate"]
            row[f"fgsm_auc_e{e}"] = v["metrics"]["auc_macro"]
        for e, v in ev.get("pgd", {}).items():
            row[f"pgd_flip_e{e}"] = v["flip_rate"]
            row[f"pgd_auc_e{e}"] = v["metrics"]["auc_macro"]
    th = r["threshold"]
    if th:
        row["loss_gap"] = th["final"]["generalization"]["loss_gap"]
        for k, v in th["final"]["threshold_attacks"].items():
            row[f"thr_{k}_auc"] = v["auc"]
    for fs, _ in HEADLINE_MIA:
        x = find_result(r, fs)
        if x:
            row[f"{fs} AUC"] = x["auc"]
            row[f"{fs} TPR@1%"] = x["tpr_at_1pct_fpr"]
            row[f"{fs} acc@median"] = x["acc_median"]
    return row


# ─── Paired bootstrap ─────────────────────────────────────────────────────────

def _auc(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _tpr_at(y, s, fpr=0.01):
    neg = np.sort(s[y == 0])[::-1]
    k = int(math.floor(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf          # strictly above the (k+1)-th negative
    return float((s[y == 1] > thr).mean())


def paired_bootstrap(y, s_c, s_d, B, seed=0):
    y = np.asarray(y).astype(int)
    s_c, s_d = np.asarray(s_c, np.float64), np.asarray(s_d, np.float64)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    yy = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    rng = np.random.RandomState(seed)
    d_auc, d_tpr = np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d_auc[b] = _auc(yy, s_d[idx]) - _auc(yy, s_c[idx])
        d_tpr[b] = _tpr_at(yy, s_d[idx]) - _tpr_at(yy, s_c[idx])
    return {"auc_c": _auc(y, s_c), "auc_d": _auc(y, s_d),
            "d_auc": _auc(y, s_d) - _auc(y, s_c),
            "d_auc_lo": float(np.percentile(d_auc, 2.5)), "d_auc_hi": float(np.percentile(d_auc, 97.5)),
            "tpr_c": _tpr_at(y, s_c), "tpr_d": _tpr_at(y, s_d),
            "d_tpr": _tpr_at(y, s_d) - _tpr_at(y, s_c),
            "d_tpr_lo": float(np.percentile(d_tpr, 2.5)), "d_tpr_hi": float(np.percentile(d_tpr, 97.5))}


def _regime(r):
    return r["meta"].get("regime", "noreg")


def pairs_of(runs):
    """(standard, adversarial) victims with the same seed AND the same regime
    (no augmentation / early stopping vs. with) — never across regimes."""
    out = []
    for d in runs.values():
        if d["training"] != "fgsm":
            continue
        for c in runs.values():
            if c["training"] == "standard" and c["seed"] == d["seed"] and _regime(c) == _regime(d):
                out.append((c, d))
    return out


def paired_table(c, d, B):
    rows = []
    sources = []
    if c["scores"] and d["scores"]:
        sources.append(("attack", c["scores"], d["scores"]))
    if c["threshold_scores"] is not None and d["threshold_scores"] is not None:
        tc = {k: v for k, v in c["threshold_scores"].items() if k != "y_true"}
        td = {k: v for k, v in d["threshold_scores"].items() if k != "y_true"}
        tc["__y__"], td["__y__"] = c["threshold_scores"]["y_true"], d["threshold_scores"]["y_true"]
        sources.append(("white-box", tc, td))
    for kind, sc, sd in sources:
        y_c, y_d = sc.get("__y__"), sd.get("__y__")
        if y_c is None or y_d is None or len(y_c) != len(y_d) or not np.array_equal(y_c, y_d):
            continue
        for key in sc:
            if key == "__y__" or key not in sd or len(sc[key]) != len(y_c):
                continue
            st = paired_bootstrap(y_c, sc[key], sd[key], B)
            rows.append({"seed": c["seed"], "C": c["run"], "D": d["run"], "source": kind,
                         "key": key, **st})
    return pd.DataFrame(rows)


# ─── Figures ──────────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_comparison(c, d, pt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_rows = sum(1 for fs, _ in HEADLINE_MIA
                 if not pt.empty and ((pt.source == "attack") & (pt.key == f"{fs}|{PRIMARY}")).any())
    fig = plt.figure(figsize=(16, max(4.4, 0.5 * n_rows + 1.6)), facecolor=SURFACE)
    # explicit placement: room for the long row labels between panels (a) and (b)
    axes = [fig.add_axes([0.045, 0.13, 0.22, 0.72]), fig.add_axes([0.50, 0.13, 0.225, 0.72]),
            fig.add_axes([0.755, 0.13, 0.225, 0.72])]
    # (a) evasion: FGSM flip rate vs ε
    ax = axes[0]
    _style(ax)
    for r, color, lab in ((c, C_STD, "Standard (C)"), (d, C_AT, "FGSM-AT (D)")):
        if not r["evasion"]:
            continue
        eps = sorted(r["evasion"]["fgsm"], key=float)
        ax.plot([float(e) for e in eps], [r["evasion"]["fgsm"][e]["flip_rate"] for e in eps],
                color=color, linewidth=2, marker="o", markersize=6, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("FGSM ε (/255, log scale)", color=INK_2, fontsize=8)
    ax.set_ylabel("Flip rate (fraction of labels changed)", color=INK_2, fontsize=8)
    ax.set_title("Evasion: FGSM flip rate (lower = more robust)", color=INK, fontsize=10, loc="left")

    # (b, c) MIA dot plots with the paired CI on D − C
    for ax, metric, title in ((axes[1], "auc", "Membership AUC (Gradient Boosting)"),
                              (axes[2], "tpr", "TPR at 1 % FPR (Gradient Boosting)")):
        _style(ax)
        labels, yc, yd, lo, hi = [], [], [], [], []
        for fs, lab in HEADLINE_MIA:
            sub = pt[(pt.source == "attack") & (pt.key == f"{fs}|{PRIMARY}")] if not pt.empty else pt
            if sub is None or len(sub) == 0:
                continue
            s = sub.iloc[0]
            labels.append(lab)
            yc.append(s[f"{metric}_c"])
            yd.append(s[f"{metric}_d"])
            lo.append(s[f"d_{metric}_lo"])
            hi.append(s[f"d_{metric}_hi"])
        ypos = np.arange(len(labels))[::-1]
        for y0, a, b, l, h in zip(ypos, yc, yd, lo, hi):
            ax.plot([a, b], [y0, y0], color=GRID, linewidth=2, zorder=1)
            # Δ label above its own row, in ink (identity is carried by the dots + legend)
            ax.annotate(f"Δ {b - a:+.3f} [{l:+.3f}, {h:+.3f}]", xy=((a + b) / 2, y0),
                        xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=INK_2)
        ax.scatter(yc, ypos, s=50, color=C_STD, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.scatter(yd, ypos, s=50, color=C_AT, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        if metric == "auc":
            ax.axvline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
        if len(labels):
            ax.set_ylim(-0.6, len(labels) - 0.2)
            vals = yc + yd
            pad = max(0.02, 0.15 * (max(vals) - min(vals)))
            ax.set_xlim(min(vals + ([0.5] if metric == "auc" else [0.0])) - pad, max(vals) + pad)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels if ax is axes[1] else [""] * len(labels), color=INK, fontsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    handles = [Line2D([], [], color=C_STD, marker="o", linestyle="", markersize=7,
                      label="Standard training (C)"),
               Line2D([], [], color=C_AT, marker="o", linestyle="", markersize=7,
                      label="FGSM adversarial training (D)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.01, f"{c['run']} vs {d['run']} — Δ = D − C with paired-bootstrap 95 % CI "
             "(same 5k + 5k evaluation images)", ha="center", color=INK_2, fontsize=8)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def gap_points(runs):
    pts = []
    for r in runs.values():
        th = r["threshold"]
        if not th:
            continue
        for ck in th.get("checkpoints", []):
            p = {"run": r["run"], "training": r["training"], "epoch": ck["epoch"],
                 **ck["generalization"]}
            for k, v in ck["threshold_attacks"].items():
                p[f"{k}_auc"] = v["auc"]
            pts.append(p)
    return pd.DataFrame(pts)


def plot_gap(points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [("loss_auc", "Loss threshold"), ("adv_loss_auc", "Adversarial loss (FGSM)"),
              ("fluct_e4_auc", "White-box fluctuation ε=4/255"), ("certainty_auc", "Confidence")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
        if key in points:
            for run, sub in points.groupby("run"):
                sub = sub.sort_values("epoch")
                color = C_AT if sub.training.iloc[0] == "fgsm" else C_STD
                ax.plot(sub.loss_gap, sub[key], color=color, linewidth=2, marker="o",
                        markersize=5, alpha=0.9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Clean loss gap (non-member − member BCE)", color=INK_2, fontsize=8)
    axes[0].set_ylabel("Membership AUC (checkpoints every 5 epochs)", color=INK_2, fontsize=8)
    handles = [Line2D([], [], color=C_STD, linewidth=2, marker="o", label="Standard training"),
               Line2D([], [], color=C_AT, linewidth=2, marker="o", label="FGSM adversarial training")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def gap_fit(points, key):
    """Linear fit of AUC on loss gap per training type, compared on the overlap."""
    if points.empty or key not in points:
        return None
    fits = {}
    for tr in ("standard", "fgsm"):
        sub = points[points.training == tr].dropna(subset=[key])
        if len(sub) < 3:
            return None
        slope, icpt = np.polyfit(sub.loss_gap, sub[key], 1)
        fits[tr] = (slope, icpt, sub.loss_gap.min(), sub.loss_gap.max())
    lo, hi = max(fits["standard"][2], fits["fgsm"][2]), min(fits["standard"][3], fits["fgsm"][3])
    if hi <= lo:
        return []
    return [{"loss_gap": g, "standard": fits["standard"][0] * g + fits["standard"][1],
             "fgsm": fits["fgsm"][0] * g + fits["fgsm"][1]} for g in np.linspace(lo, hi, 4)]


# ─── Paper-format section (same metrics as the submitted paper) ──────────────

def _bar_pair(ax, labels, values, colors, chance=None, fmt="{:.1%}"):
    xs = np.arange(len(values))
    ax.bar(xs, values, width=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    for x0, v in zip(xs, values):
        ax.annotate(fmt.format(v), xy=(x0, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)
    if chance is not None:
        ax.axhline(chance, color=INK_2, linewidth=0.8, linestyle=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=INK, fontsize=8)


def plot_paper_figures(c, d, out_dir):
    """Fig. 6 (flip rate vs normalised ε) and Fig. 7 (robustness–privacy trade-off)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ec, ed = c["evasion"], d["evasion"]
    if not (ec and ed and ec.get("paper_fgsm") and ed.get("paper_fgsm")):
        return []
    made = []
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    _style(ax)
    for r, color, lab in ((c, C_STD, "Standard (C)"), (d, C_AT, "FGSM-AT (D)")):
        eps = [0.0] + [float(e) for e in PAPER_EPS]
        flips = [0.0] + [r["evasion"]["paper_fgsm"][e]["flip_rate"] for e in PAPER_EPS]
        ax.plot(eps, flips, color=color, linewidth=2, marker="o", markersize=6, label=lab)
    ax.set_xlabel("FGSM perturbation ε (normalised units, as in the paper)", color=INK_2, fontsize=8)
    ax.set_ylabel("Flip rate (fraction of flipped predictions)", color=INK_2, fontsize=8)
    ax.set_title("Paper Fig. 6 — evasion flip rate vs ε", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    p = os.path.join(out_dir, f"paper_fig6_{d['run']}.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    made.append(os.path.basename(p))

    best = []
    for r in (c, d):
        accs = [x["accuracy"] for _, fs in PAPER_STRATEGIES
                for x in [find_result(r, fs, PRIMARY)] if x]
        best.append(max(accs) if accs else float("nan"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    _bar_pair(axes[0], ["Standard (C)", "FGSM-AT (D)"],
              [ec["paper_fgsm"]["0.02"]["flip_rate"], ed["paper_fgsm"]["0.02"]["flip_rate"]],
              [C_STD, C_AT])
    axes[0].set_title("Evasion: flip rate at ε = 0.02", color=INK, fontsize=10, loc="left")
    _bar_pair(axes[1], ["Standard (C)", "FGSM-AT (D)"], best, [C_STD, C_AT], chance=0.5)
    axes[1].set_title("Privacy: best MIA accuracy (GB, paper strategies)", color=INK,
                      fontsize=10, loc="left")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    p = os.path.join(out_dir, f"paper_fig7_{d['run']}.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    made.append(os.path.basename(p))
    return made


def paper_section(runs, pairs, out_dir):
    L = ["## Paper-format results (the same metrics as the submitted paper)\n",
         "Feature strategies as named in the paper: **Original MIA** = shadow-model MIA on "
         "confidence scores, **+ Variance** = + bootstrapped variance-of-max (BCVA), "
         "**Fluctuation** = AFMIA with the paper's features. Evasion uses the paper's ε grid "
         "(normalised units, clamp ±3, all 33,412 non-members).\n"]

    L.append("### Training and generalisation (paper §IV-B3 and README key findings)\n")
    L.append("| Run | Epochs trained | Selected epoch | Train acc | Val acc | Memorization gap | "
             "Clean test AUC | Avg max-conf members | Avg max-conf non-members | Avg per-class "
             "fluctuation members (Eq. 7) | non-members |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs.values():
        m = r["meta"]
        g = (r["threshold"] or {}).get("final", {}).get("generalization", {})
        pf = g.get("paper_fluct") or {}
        pct = lambda v: "—" if v is None else f"{v:.2f} %"   # noqa: E731
        mg = m.get("memorization_gap")
        L.append(f"| `{r['run']}` | {m.get('epochs_trained', '—')} | {m.get('selected_epoch', '—')} | "
                 f"{f4(m.get('final_train_acc'))} | {f4(m.get('final_val_acc'))} | "
                 f"{'—' if mg is None else f'{100 * mg:+.2f} %'} | "
                 f"{f4((r['evasion'] or {}).get('clean', {}).get('auc_macro'))} | "
                 f"{f4(g.get('avg_max_conf_members'))} | {f4(g.get('avg_max_conf_nonmembers'))} | "
                 f"{pct(pf.get('avg_members_pct'))} | {pct(pf.get('avg_nonmembers_pct'))} |")
    L.append("\nEq. 7 fluctuation: FGSM ε = 0.01 (normalised) on the true labels, per-class "
             "mean |p(x) − p(x_adv)| on the 5k + 5k evaluation images, averaged over the 15 classes "
             "(paper: 2.67 % members vs 7.41 % non-members on its adversarial model).\n")

    L.append("### Table I — evasion: classification AUC and flip rate under FGSM\n")
    done = set()
    for c, d in pairs:
        ec, ed = c["evasion"], d["evasion"]
        if not (ec and ed and ec.get("paper_fgsm") and ed.get("paper_fgsm")):
            continue
        L.append(f"`{c['run']}` (baseline) vs `{d['run']}` (adversarial)\n")
        L.append("| ε | Baseline AUC | Adversarial AUC | Baseline flip | Adversarial flip | "
                 "Robustness gain (Δ AUC) |\n|---|---|---|---|---|---|")
        L.append(f"| 0.000 (clean) | {ec['clean']['auc_macro']:.4f} | {ed['clean']['auc_macro']:.4f} "
                 f"| — | — | {ed['clean']['auc_macro'] - ec['clean']['auc_macro']:+.4f} |")
        for e in PAPER_EPS:
            a, b = ec["paper_fgsm"][e], ed["paper_fgsm"][e]
            L.append(f"| {float(e):.3f} | {a['metrics']['auc_macro']:.4f} | {b['metrics']['auc_macro']:.4f} "
                     f"| {100 * a['flip_rate']:.1f} % | {100 * b['flip_rate']:.1f} % | "
                     f"{b['metrics']['auc_macro'] - a['metrics']['auc_macro']:+.4f} |")
        L.append("")
        done.update([c["run"], d["run"]])
        figs = plot_paper_figures(c, d, out_dir)
        L.extend(f"![{f}]({f})\n" for f in figs)
    for r in runs.values():
        ev = r["evasion"]
        if r["run"] in done or not ev or not ev.get("paper_fgsm"):
            continue
        L.append(f"`{r['run']}` (single victim in this folder)\n")
        L.append("| ε | AUC | Flip rate |\n|---|---|---|")
        L.append(f"| 0.000 (clean) | {ev['clean']['auc_macro']:.4f} | — |")
        for e in PAPER_EPS:
            a = ev["paper_fgsm"][e]
            L.append(f"| {float(e):.3f} | {a['metrics']['auc_macro']:.4f} | {100 * a['flip_rate']:.1f} % |")
        L.append("")

    L.append("### Tables II / III — MIA results per victim (Acc / Prec / Rec / F1)\n")
    for r in runs.values():
        if not r["attacks"]:
            continue
        L.append(f"**`{r['run']}`** ({'adversarial' if r['training'] == 'fgsm' else 'baseline'} "
                 "victim)\n")
        L.append("| Feature strategy | Attack classifier | Acc | Prec | Rec | F1 |\n|---|---|---|---|---|---|")
        for paper_name, fs in PAPER_STRATEGIES:
            for clf in CLASSIFIERS:
                x = find_result(r, fs, clf)
                if x:
                    L.append(f"| {paper_name} | {clf} | {x['accuracy']:.4f} | {x['precision']:.4f} | "
                             f"{x['recall']:.4f} | {x['f1']:.4f} |")
        L.append("")
    if pairs:
        L.append("### Fig. 7 numbers — the robustness–privacy trade-off\n")
        L.append("| Victim | Flip rate ε = 0.02 | " + " | ".join(f"{p} acc (GB)" for p, _ in PAPER_STRATEGIES)
                 + " | Best MIA acc (GB) |")
        L.append("|---|---|" + "---|" * (len(PAPER_STRATEGIES) + 1))
        for r in {x["run"]: x for c, d in pairs for x in (c, d)}.values():
            ev = r["evasion"] or {}
            flip = ev.get("paper_fgsm", {}).get("0.02", {}).get("flip_rate")
            accs = [find_result(r, fs, PRIMARY) for _, fs in PAPER_STRATEGIES]
            vals = [x["accuracy"] for x in accs if x]
            best = f"{max(vals):.4f}" if vals else "—"
            flip_txt = "—" if flip is None else f"{100 * flip:.1f} %"
            L.append(f"| `{r['run']}` | {flip_txt} | "
                     + " | ".join("—" if x is None else f"{x['accuracy']:.4f}" for x in accs)
                     + f" | {best} |")
        L.append("")
    return L


# ─── Report ───────────────────────────────────────────────────────────────────

def f4(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def ci(st, m):
    return f"{st[f'd_{m}']:+.4f} [{st[f'd_{m}_lo']:+.4f}, {st[f'd_{m}_hi']:+.4f}]"


def verdict(st, m, expect_positive=True):
    lo, hi = st[f"d_{m}_lo"], st[f"d_{m}_hi"]
    if (lo > 0 and expect_positive) or (hi < 0 and not expect_positive):
        return "**yes**"
    if (hi < 0 and expect_positive) or (lo > 0 and not expect_positive):
        return "opposite"
    return "n.s."


def headline(c, d, pt):
    L = [f"### Seed {c['seed']}: `{c['run']}` (C) vs `{d['run']}` (D)\n",
         "| Attack | Metric | C standard | D FGSM-AT | D − C [95 % CI] | Expected | Holds? |",
         "|---|---|---|---|---|---|---|"]
    ec, ed = c["evasion"], d["evasion"]
    if ec and ed:
        for kind, key, lab in (("fgsm", "flip_rate", "flip rate"), ("fgsm", "auc", "adv. AUC"),
                               ("pgd", "flip_rate", "flip rate"), ("pgd", "auc", "adv. AUC")):
            a, b = ec.get(kind, {}).get(HEAD_EPS), ed.get(kind, {}).get(HEAD_EPS)
            if not a or not b:
                continue
            va = a["flip_rate"] if key == "flip_rate" else a["metrics"]["auc_macro"]
            vb = b["flip_rate"] if key == "flip_rate" else b["metrics"]["auc_macro"]
            want_lower = key == "flip_rate"
            ok = (vb < va) if want_lower else (vb > va)
            L.append(f"| Evasion ({kind.upper()}{'-10' if kind == 'pgd' else ''}, ε={HEAD_EPS}/255) "
                     f"| {lab} | {va:.4f} | {vb:.4f} | {vb - va:+.4f} | "
                     f"D {'<' if want_lower else '>'} C | {'**yes**' if ok else 'no'} |")
        L.append(f"| Clean utility | test AUC | {ec['clean']['auc_macro']:.4f} | "
                 f"{ed['clean']['auc_macro']:.4f} | "
                 f"{ed['clean']['auc_macro'] - ec['clean']['auc_macro']:+.4f} | — | — |")
    for fs, lab in HEADLINE_MIA:
        sub = pt[(pt.source == "attack") & (pt.key == f"{fs}|{PRIMARY}")] if not pt.empty else pt
        if sub is None or len(sub) == 0:
            continue
        st = sub.iloc[0]
        L.append(f"| {lab} | AUC | {st.auc_c:.4f} | {st.auc_d:.4f} | {ci(st, 'auc')} | D > C | "
                 f"{verdict(st, 'auc')} |")
        L.append(f"| {lab} | TPR@1 % FPR | {st.tpr_c:.4f} | {st.tpr_d:.4f} | {ci(st, 'tpr')} | "
                 f"D > C | {verdict(st, 'tpr')} |")
        xc, xd = find_result(c, fs), find_result(d, fs)
        if xc and xd:
            L.append(f"| {lab} | acc @ median | {xc['acc_median']:.4f} | {xd['acc_median']:.4f} | "
                     f"{xd['acc_median'] - xc['acc_median']:+.4f} | D > C | — |")
            L.append(f"| {lab} | acc @ 0.5 (paper metric) | {xc['accuracy']:.4f} | "
                     f"{xd['accuracy']:.4f} | {xd['accuracy'] - xc['accuracy']:+.4f} | D > C | — |")
    L.append("\n`yes` = the 95 % CI of D − C excludes 0 in the expected direction; `n.s.` = it "
             "contains 0; `opposite` = significant in the other direction. Evasion rows are "
             "single numbers on 33,412 (FGSM) / 5,000 (PGD) images.\n")
    return L


def write_report(R, runs, pts_all, attack_df, gap_pts, out_dir, B):
    L = ["# Matched experiment v2 — results\n",
         "Two victims trained with the same script, data split, seed and attack pipeline, both "
         "**without** augmentation or early stopping (last of 35 epochs): **C** standard "
         "training and **D** FGSM adversarial training (FGSM-RS by default). Members = images the "
         "victim trained on; non-members = patient-disjoint split; the 5k + 5k evaluation images "
         "are identical for C and D (paired design). Primary MIA metrics are AUC and TPR at low "
         "FPR; `acc @ 0.5` is the paper's metric (threshold learned on shadow data), "
         "`acc @ median` calls the top half of the scores members (label-free calibration on the "
         "balanced evaluation set).\n"]

    L.append("## Victims\n")
    L.append("| Run | Training | Method | ε (/255) | α | Train acc | Val AUC | FGSM val AUC | "
             "PGD val AUC | Max epoch swing of FGSM val AUC | Clean test AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs.values():
        row = run_row(r)
        L.append(f"| `{r['run']}` | {r['training']} | {row['method'] or '—'} | "
                 f"{row['train_eps255'] if row['train_eps255'] is not None else '—'} | "
                 f"{row['alpha'] if row['alpha'] is not None else '—'} | {f4(row['train_acc'])} | "
                 f"{f4(row['val_auc'])} | {f4(row['fgsm_val_auc'])} | {f4(row.get('pgd_val_auc'))} | "
                 f"{f4(row.get('max_epoch_swing_fgsm_val_auc'))} | {f4(row.get('test_auc_clean'))} |")
    L.append("\nA large epoch-to-epoch swing of the FGSM val AUC, or FGSM val AUC far above PGD val "
             "AUC, indicates catastrophic overfitting (gradient masking) of single-step AT.\n")
    imported = [r["run"] for r in runs.values() if r["meta"].get("imported_from")]
    if imported:
        L.append(f"Imported from a v1 output (same recipe, trained by v1 code): "
                 f"{', '.join(imported)}.\n")

    pairs = pairs_of(runs)
    L.extend(paper_section(runs, pairs, out_dir))
    L.append("## Headline — the three attacks, C vs D\n")
    if not pairs:
        L.append("_Needs both victims of a seed in this results folder — merge the two Kaggle "
                 "outputs (kaggle/merge_kaggle_results.py)._\n")
    for c, d in pairs:
        pt = pts_all[(pts_all.C == c["run"]) & (pts_all.D == d["run"])] if not pts_all.empty \
            else pts_all
        L.extend(headline(c, d, pt))
        fig = f"attack_comparison_{d['run']}.png"
        if not pt.empty:
            L.append(f"![comparison]({fig})\n")

    if len({c["seed"] for c, _ in pairs}) >= 2:
        L.append("## Across seeds (mean of per-seed D − C AUC, Gradient Boosting)\n")
        L.append("| Attack | n seeds | mean Δ AUC | min | max |\n|---|---|---|---|---|")
        for fs, lab in HEADLINE_MIA:
            sub = pts_all[(pts_all.source == "attack") & (pts_all.key == f"{fs}|{PRIMARY}")]
            if len(sub):
                L.append(f"| {lab} | {sub.seed.nunique()} | {sub.d_auc.mean():+.4f} | "
                         f"{sub.d_auc.min():+.4f} | {sub.d_auc.max():+.4f} |")
        L.append("")

    if not pts_all.empty:
        L.append("## Paired comparison — every score (D − C, paired bootstrap, "
                 f"B = {B})\n")
        L.append("| Seed | Source | Score | AUC C | AUC D | Δ AUC [95 % CI] | TPR@1% C | TPR@1% D | "
                 "Δ TPR@1% [95 % CI] |\n|---|---|---|---|---|---|---|---|---|")
        for _, s in pts_all.iterrows():
            key = (THRESH_LABELS.get(s.key, s.key) if s.source == "white-box" else s.key).replace("|", " / ")
            L.append(f"| {s.seed} | {s.source} | {key} | {s.auc_c:.4f} | {s.auc_d:.4f} | "
                     f"{ci(s, 'auc')} | {s.tpr_c:.4f} | {s.tpr_d:.4f} | {ci(s, 'tpr')} |")
        L.append("\n`white-box` rows are the reference threshold attacks of evaluate_victim.py "
                 "(full model access — an upper bound, not part of the black-box comparison).\n")

    L.append("## All attack results per victim\n")
    if not attack_df.empty:
        L.append("| Run | Feature set | Classifier | AUC | TPR@1% | TPR@0.1% | Acc@0.5 | Acc@median | "
                 "Best-thr acc | Prec | Rec | F1 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, x in attack_df.iterrows():
            L.append(f"| `{x.run}` | {x.feature_set} | {x.classifier} | {x.auc:.4f} | "
                     f"{x.tpr_at_1pct_fpr:.4f} | {x['tpr_at_0.1pct_fpr']:.4f} | {x.accuracy:.4f} | "
                     f"{x.acc_median:.4f} | {x.best_threshold_acc:.4f} | {x.precision:.4f} | "
                     f"{x.recall:.4f} | {x.f1:.4f} |")
        L.append("\n`Best-thr acc` picks the threshold on the evaluation labels — optimistic.\n")
    else:
        L.append("_No attack results yet._\n")

    L.append("## Classifier-free scores on the victim (from the attack runs)\n")
    any_thr = False
    for r in runs.values():
        for a, res in r["attacks"].items():
            rows = res.get("threshold_scores") or []
            if not rows:
                continue
            if not any_thr:
                L.append("| Run | Attack run | Score | AUC on victim | TPR@1% | Acc@median | "
                         "AUC on attacker's shadow rows |\n|---|---|---|---|---|---|---|")
                any_thr = True
            for t in rows:
                L.append(f"| `{r['run']}` | {a} | {t['key']} | {t['auc']:.4f} | "
                         f"{t['tpr_at_1pct_fpr']:.4f} | {t['acc_median']:.4f} | "
                         f"{t['shadow_row_auc']:.4f} |")
    L.append("" if any_thr else "_None yet._\n")
    L.append("\nA score with AUC > 0.5 on the shadow rows but < 0.5 on a victim is a signal "
             "the attacker learned that points the other way on that victim.\n")

    L.append("## Evasion detail\n")
    for r in runs.values():
        ev = r["evasion"]
        if not ev:
            continue
        L.append(f"**`{r['run']}`** — clean AUC {ev['clean']['auc_macro']:.4f}\n")
        L.append("| Attack | ε (/255) | Adv. AUC | Flip rate |\n|---|---|---|---|")
        for kind in ("fgsm", "pgd"):
            for e in sorted(ev.get(kind, {}), key=float):
                v = ev[kind][e]
                L.append(f"| {kind.upper()} | {e} | {v['metrics']['auc_macro']:.4f} | "
                         f"{v['flip_rate']:.4f} |")
        L.append("")

    L.append("## Gap-matched analysis (checkpoints every 5 epochs)\n")
    if not gap_pts.empty:
        for key in ("loss_auc", "adv_loss_auc", "fluct_e4_auc"):
            rows = gap_fit(gap_pts, key)
            if rows is None:
                continue
            L.append(f"**{key}** — linear fit per training type, evaluated where the loss gaps "
                     "overlap (descriptive; checkpoints of a run are correlated)\n")
            if rows:
                L.append("| loss gap | standard | FGSM-AT | FGSM − standard |\n|---|---|---|---|")
                for x in rows:
                    L.append(f"| {x['loss_gap']:.4f} | {x['standard']:.4f} | {x['fgsm']:.4f} | "
                             f"{x['fgsm'] - x['standard']:+.4f} |")
            else:
                L.append("_No overlapping gap range._")
            L.append("")
        L.append("![gap-matched](gap_matched.png)\n")
    else:
        L.append("_No checkpoint evaluations yet._\n")

    L.append("## Missing outputs\n")
    miss = [f"- `{r['run']}`: {m}" for r in runs.values() for m in r["missing"]]
    L.extend(miss or ["- none"])
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def run(R, bootstrap=1000):
    """Build summary/ for results folder R.  Returns (runs, paired table) — also
    used by Matched_Experiments_v3/aggregate_lab.py."""
    out_dir = os.path.join(R, "summary")
    os.makedirs(out_dir, exist_ok=True)

    runs = collect(R)
    if not runs:
        print("[AGG] no victims found yet.")
        return runs, pd.DataFrame()
    pd.DataFrame([run_row(r) for r in runs.values()]).to_csv(
        os.path.join(out_dir, "per_run.csv"), index=False)
    attack_df = pd.DataFrame([x for r in runs.values() for x in attack_rows(r)])
    if not attack_df.empty:
        attack_df.to_csv(os.path.join(out_dir, "attack_results.csv"), index=False)

    pts = [paired_table(c, d, bootstrap) for c, d in pairs_of(runs)]
    pts_all = pd.concat([x for x in pts if not x.empty], ignore_index=True) \
        if any(not x.empty for x in pts) else pd.DataFrame()
    if not pts_all.empty:
        pts_all.to_csv(os.path.join(out_dir, "paired.csv"), index=False)
        for c, d in pairs_of(runs):
            sub = pts_all[(pts_all.C == c["run"]) & (pts_all.D == d["run"])]
            if not sub.empty:
                plot_comparison(c, d, sub, os.path.join(out_dir, f"attack_comparison_{d['run']}.png"))

    gap_pts = gap_points(runs)
    if not gap_pts.empty:
        gap_pts.to_csv(os.path.join(out_dir, "gap_matched_points.csv"), index=False)
        plot_gap(gap_pts, os.path.join(out_dir, "gap_matched.png"))
    write_report(R, runs, pts_all, attack_df, gap_pts, out_dir, bootstrap)
    print(f"[AGG] {len(runs)} victims, {len(pairs_of(runs))} C/D pair(s) → {out_dir}/REPORT.md")
    return runs, pts_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=os.path.join(v2.V2_DIR, "results_v2"))
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    run(args.results_dir, args.bootstrap)


if __name__ == "__main__":
    main()
