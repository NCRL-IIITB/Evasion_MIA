"""
aggregate_results.py
====================
Collect every per-victim result of the matched experiment into tables,
paired-seed factorial contrasts, and the gap-matched analysis.

Reads   <results>/victims/*/model_meta.json
        <results>/eval/*/{evasion,threshold_mia}.json
        <results>/attacks/{shadow_mia,afmia}/*/results.json
Writes  <results>/summary/
          per_run.csv               one row per victim (all headline metrics)
          cell_summary.csv / .md    mean ± std over seeds per cell
          contrasts.csv / .md       paired-seed effects with 95 % CIs
          gap_matched_points.csv    every checkpoint: generalisation gap vs leakage
          gap_matched_fit.md        leakage at equal generalisation gap (std vs FGSM-AT)
          gap_matched.png           threshold-MIA AUC vs clean loss gap
          cell_comparison.png       headline metrics per cell, one dot per seed
          REPORT.md                 everything above in one file

Safe to run at any time — missing runs are listed, never guessed.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_common as mc

CELL_ORDER = ["std_reg", "fgsm_reg", "fgsm_reg_robustES", "std_noreg", "fgsm_noreg"]
PRIMARY_CLASSIFIER = "Gradient Boosting"   # the paper's headline classifier

# Paired contrasts: (name, plus-cells, minus-cells)
CONTRASTS = [
    ("AT effect | regularised (B − A)",           ["fgsm_reg"],   ["std_reg"]),
    ("AT effect | unregularised (D − C)",         ["fgsm_noreg"], ["std_noreg"]),
    ("Removing aug+ES | standard (C − A)",        ["std_noreg"],  ["std_reg"]),
    ("Removing aug+ES | AT (D − B)",              ["fgsm_noreg"], ["fgsm_reg"]),
    ("Interaction (D − C) − (B − A)",             ["fgsm_noreg", "std_reg"], ["std_noreg", "fgsm_reg"]),
    ("Robust-ES vs clean-ES | AT (B2 − B)",       ["fgsm_reg_robustES"], ["fgsm_reg"]),
    ("Paper comparison (D − A)",                  ["fgsm_noreg"], ["std_reg"]),
]

KEY_METRICS = [
    ("shadow_mia_gb_acc",       "Shadow-MIA acc (GB)"),
    ("shadow_mia_gb_auc",       "Shadow-MIA AUC (GB)"),
    ("shadow_mia_bcva_gb_acc",  "Shadow-MIA+BCVA acc (GB)"),
    ("afmia_gb_acc",            "AFMIA acc (GB)"),
    ("afmia_gb_auc",            "AFMIA AUC (GB)"),
    ("afmia_gb_tpr1",           "AFMIA TPR@1%FPR (GB)"),
    ("afmia_bcva_gb_acc",       "AFMIA+BCVA acc (GB)"),
    ("thr_loss_auc",            "Loss-threshold AUC"),
    ("thr_fluct_wb_auc",        "White-box fluctuation AUC"),
    ("thr_adv_loss_auc",        "Adv-loss-threshold AUC"),
    ("loss_gap",                "Clean loss gap (NM − M)"),
    ("memorization_gap",        "Train−val acc gap"),
    ("test_auc_clean",          "Clean test AUC"),
    ("flip_eps0.02",            "FGSM flip rate ε=0.02"),
    ("adv_auc_eps0.02",         "FGSM test AUC ε=0.02"),
]

# Paper numbers for reference (Tables I/II, Fig. 5; original 1-seed runs)
PAPER_REFERENCE = [
    ("Shadow-MIA acc (GB)", "0.5000", "0.6831"),
    ("Shadow-MIA+BCVA acc (GB)", "0.5000", "0.6831"),
    ("AFMIA acc (GB)", "0.5016", "0.6730"),
    ("AFMIA+BCVA acc (GB)", "0.5004", "0.6744"),
    ("Clean test AUC", "0.8278", "0.7516"),
    ("FGSM flip rate ε=0.02", "0.261", "0.039"),
]

COLORS = {"standard": "#2a78d6", "fgsm": "#eb6834"}     # validated categorical slots 1–2
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _attack_fields(res, prefix):
    """Flatten one results.json → {prefix_gb_acc, …, prefix_best_acc}."""
    out = {}
    by_set = defaultdict(list)
    for r in res["results"]:
        by_set[r["feature_set"]].append(r)
    for fs, rows in by_set.items():
        p = prefix + ("_bcva" if "BCVA" in fs else "")
        gb = next(r for r in rows if r["classifier"] == PRIMARY_CLASSIFIER)
        out.update({f"{p}_gb_acc": gb["accuracy"], f"{p}_gb_f1": gb["f1"],
                    f"{p}_gb_prec": gb["precision"], f"{p}_gb_rec": gb["recall"],
                    f"{p}_gb_auc": gb["auc"], f"{p}_gb_tpr1": gb["tpr_at_1pct_fpr"],
                    f"{p}_gb_tpr01": gb["tpr_at_0.1pct_fpr"],
                    f"{p}_best_acc": max(r["accuracy"] for r in rows),
                    f"{p}_best_auc": max(r["auc"] for r in rows)})
    out[f"{prefix}_conf_gap"] = res.get("conf_gap")
    return out


def collect(results_dir):
    rows, points, missing = [], [], []
    for meta_path in sorted(glob.glob(os.path.join(results_dir, "victims", "*", "model_meta.json"))):
        run = os.path.basename(os.path.dirname(meta_path))
        cell, seed = run.rsplit("_seed", 1)
        if cell not in mc.CELLS:
            continue
        meta = _load(meta_path)
        row = {"run": run, "cell": cell, "tag": mc.CELLS[cell]["tag"], "seed": int(seed),
               "training": mc.CELLS[cell]["training"], "regime": mc.CELLS[cell]["regime"],
               "selected_epoch": meta.get("selected_epoch"),
               "early_stop_epoch": meta.get("early_stop_epoch"),
               "val_auc": meta.get("final_val_auc"), "adv_val_auc": meta.get("final_adv_val_auc"),
               "train_acc": meta.get("final_train_acc"),
               "memorization_gap": meta.get("memorization_gap")}

        ev_dir = os.path.join(results_dir, "eval", run)
        ev_path, th_path = os.path.join(ev_dir, "evasion.json"), os.path.join(ev_dir, "threshold_mia.json")
        if os.path.exists(ev_path):
            ev = _load(ev_path)
            row["test_auc_clean"] = ev["clean"]["auc_macro"]
            for eps, v in ev["adversarial"].items():      # keys like flip_eps0.02
                row[f"flip_eps{eps}"] = v["flip_rate"]
                row[f"adv_auc_eps{eps}"] = v["metrics"]["auc_macro"]
        else:
            missing.append(f"{run}: eval/evasion.json")
        if os.path.exists(th_path):
            th = _load(th_path)
            g, a = th["final"]["generalization"], th["final"]["threshold_attacks"]
            row.update({"loss_gap": g["loss_gap"], "auc_gap": g["auc_gap"],
                        "adv_loss_gap": g["adv_loss_gap"],
                        "member_fluct": g["member_fluct"], "nonmember_fluct": g["nonmember_fluct"]})
            for name, m in a.items():
                row[f"thr_{name}_auc"] = m["auc"]
                row[f"thr_{name}_tpr1"] = m["tpr_at_1pct_fpr"]
            if cell != "fgsm_reg_robustES":      # B2 shares B's trajectory
                for ck in th["checkpoints"]:
                    pt = {"run": run, "cell": cell, "seed": int(seed),
                          "training": row["training"], "regime": row["regime"],
                          "epoch": ck["epoch"], **ck["generalization"]}
                    for name, m in ck["threshold_attacks"].items():
                        pt[f"{name}_auc"] = m["auc"]
                        pt[f"{name}_tpr1"] = m["tpr_at_1pct_fpr"]
                    points.append(pt)
        else:
            missing.append(f"{run}: eval/threshold_mia.json")

        for attack, prefix in (("shadow_mia", "shadow_mia"), ("afmia", "afmia")):
            rp = os.path.join(results_dir, "attacks", attack, run, "results.json")
            if os.path.exists(rp):
                row.update(_attack_fields(_load(rp), prefix))
            else:
                missing.append(f"{run}: attacks/{attack}/results.json")
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(points), missing


# ─── Statistics ───────────────────────────────────────────────────────────────

def _t975(df):
    from scipy.stats import t
    return float(t.ppf(0.975, df))


def mean_ci(values):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   dtype=float)
    n = len(v)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    m = float(v.mean())
    if n == 1:
        return dict(n=1, mean=m, std=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    s = float(v.std(ddof=1))
    h = _t975(n - 1) * s / math.sqrt(n)
    return dict(n=n, mean=m, std=s, ci_lo=m - h, ci_hi=m + h)


def cell_summary(df):
    out = []
    for cell in CELL_ORDER:
        sub = df[df.cell == cell]
        if sub.empty:
            continue
        for key, label in KEY_METRICS:
            if key not in sub:
                continue
            st = mean_ci(sub[key].tolist())
            out.append({"cell": cell, "tag": mc.CELLS[cell]["tag"], "metric": key, "label": label,
                        **st, "per_seed": ";".join(f"{s}:{v:.4f}" for s, v in
                                                   zip(sub.seed, sub[key]) if pd.notna(v))})
    return pd.DataFrame(out)


def contrasts(df):
    out = []
    for name, plus, minus in CONTRASTS:
        cells = plus + minus
        seeds = sorted(set.intersection(*[set(df[df.cell == c].seed) for c in cells])) \
            if all((df.cell == c).any() for c in cells) else []
        for key, label in KEY_METRICS:
            if key not in df or not seeds:
                continue
            diffs = []
            for s in seeds:
                vals = {c: df[(df.cell == c) & (df.seed == s)][key].iloc[0] for c in cells}
                if any(pd.isna(v) for v in vals.values()):
                    continue
                diffs.append(sum(vals[c] for c in plus) - sum(vals[c] for c in minus))
            st = mean_ci(diffs)
            if st["n"] == 0:
                continue
            excludes_zero = (st["n"] > 1 and (st["ci_lo"] > 0 or st["ci_hi"] < 0))
            out.append({"contrast": name, "metric": key, "label": label, **st,
                        "ci_excludes_0": excludes_zero,
                        "per_seed": ";".join(f"{d:+.4f}" for d in diffs)})
    return pd.DataFrame(out)


def gap_matched_fit(points, attack_key):
    """Linear fit of attack AUC on clean loss gap, per training type, compared
    on the overlapping gap range (descriptive: checkpoints are not independent)."""
    if points.empty or attack_key not in points:
        return None
    fits = {}
    for tr in ("standard", "fgsm"):
        sub = points[points.training == tr]
        if len(sub) < 3:
            return None
        slope, intercept = np.polyfit(sub.loss_gap, sub[attack_key], 1)
        fits[tr] = (slope, intercept, sub.loss_gap.min(), sub.loss_gap.max(), len(sub))
    lo = max(fits["standard"][2], fits["fgsm"][2])
    hi = min(fits["standard"][3], fits["fgsm"][3])
    grid = np.linspace(lo, hi, 5) if hi > lo else np.array([])
    rows = [{"loss_gap": g,
             "standard": fits["standard"][0] * g + fits["standard"][1],
             "fgsm": fits["fgsm"][0] * g + fits["fgsm"][1]} for g in grid]
    for r in rows:
        r["fgsm_minus_standard"] = r["fgsm"] - r["standard"]
    return fits, rows


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


def plot_gap_matched(points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    attacks = [("loss_auc", "Loss threshold"), ("fluct_wb_auc", "White-box fluctuation"),
               ("adv_loss_auc", "Adversarial-loss threshold"), ("confidence_auc", "Confidence")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, attacks):
        _style(ax)
        ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
        for run, sub in points.groupby("run"):
            sub = sub.sort_values("epoch")
            tr, rg = sub.training.iloc[0], sub.regime.iloc[0]
            ax.plot(sub.loss_gap, sub[key], color=COLORS[tr], linewidth=1.5,
                    linestyle="-" if rg == "reg" else "--",
                    marker="o" if rg == "reg" else "s", markersize=5,
                    markerfacecolor=COLORS[tr] if rg == "reg" else SURFACE,
                    markeredgecolor=COLORS[tr], markeredgewidth=1.2, alpha=0.9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Clean loss gap (non-member − member BCE)", color=INK_2, fontsize=8)
    axes[0].set_ylabel("Membership AUC", color=INK_2, fontsize=9)
    axes[0].annotate("chance", xy=(0, 0.5), xycoords=("axes fraction", "data"),
                     xytext=(3, 3), textcoords="offset points", color=INK_2, fontsize=7)
    handles = [
        Line2D([], [], color=COLORS["standard"], linewidth=1.5, label="Standard training"),
        Line2D([], [], color=COLORS["fgsm"], linewidth=1.5, label="FGSM adversarial training"),
        Line2D([], [], color=INK_2, linestyle="-", marker="o", markersize=5,
               label="Aug + early stopping (checkpoints)"),
        Line2D([], [], color=INK_2, linestyle="--", marker="s", markersize=5,
               markerfacecolor=SURFACE, label="No aug, 35 epochs (checkpoints)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_cell_comparison(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("shadow_mia_gb_acc", "Shadow-model MIA accuracy (GB)"),
               ("afmia_gb_acc", "AFMIA accuracy (GB)"),
               ("thr_loss_auc", "Loss-threshold MIA AUC"),
               ("flip_eps0.02", "FGSM flip rate, ε = 0.02")]
    cells = [c for c in CELL_ORDER if (df.cell == c).any()]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, metrics):
        _style(ax)
        if key not in df:
            ax.set_title(f"{title} (no data yet)", color=INK_2, fontsize=9, loc="left")
            continue
        for i, cell in enumerate(cells):
            sub = df[df.cell == cell][key].dropna()
            color = COLORS[mc.CELLS[cell]["training"]]
            filled = mc.CELLS[cell]["regime"] == "reg"
            jitter = np.linspace(-0.12, 0.12, len(sub)) if len(sub) > 1 else [0.0]
            ax.scatter(i + np.asarray(jitter), sub, s=36, color=color if filled else SURFACE,
                       edgecolors=color, linewidths=1.4, zorder=3,
                       marker="o" if filled else "s")
            if len(sub):
                ax.hlines(sub.mean(), i - 0.25, i + 0.25, color=INK, linewidth=1.5, zorder=4)
        if "acc" in key or "auc" in key:
            ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([mc.CELLS[c]["tag"] for c in cells], color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.text(0.5, 0.005, "A std+reg · B FGSM+reg · B2 FGSM+reg (robust ES) · C std no-reg · "
             "D FGSM no-reg   —   blue = standard, orange = FGSM-AT, filled = aug+ES, "
             "hollow = no aug/ES, bar = mean, dot = seed",
             ha="center", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ─── Report ───────────────────────────────────────────────────────────────────

def _fmt(st, digits=4):
    if st["n"] == 0 or pd.isna(st["mean"]):
        return "—"
    if st["n"] == 1:
        return f"{st['mean']:.{digits}f} (n=1)"
    return f"{st['mean']:.{digits}f} ± {st['std']:.{digits}f}"


def write_report(results_dir, df, cs, ct, points, fit_rows, missing, out_dir):
    L = []
    L.append("# Matched-control experiment — results\n")
    L.append("Design: 2 × 2 (training loss × regularisation) with the same script, data split, "
             "seed set and attack pipeline for every cell; B2 = cell B selected by adversarial "
             "val AUC. Members = images the victim trained on (validation hold-out excluded); "
             "non-members = patient-disjoint split. Attack seed = victim seed (paired design).\n")
    seeds = sorted(int(s) for s in df.seed.unique()) if not df.empty else []
    L.append(f"Runs found: {len(df)} victims, seeds {seeds}.\n")
    L.append("| Cell | Description |\n|---|---|")
    for c in CELL_ORDER:
        L.append(f"| {mc.CELLS[c]['tag']} `{c}` | {mc.CELLS[c]['desc']} |")
    L.append("")

    if not cs.empty:
        L.append("## Per-cell results (mean ± std over seeds)\n")
        present = [c for c in CELL_ORDER if (cs.cell == c).any()]
        L.append("| Metric | " + " | ".join(mc.CELLS[c]["tag"] for c in present) + " |")
        L.append("|---|" + "---|" * len(present))
        for key, label in KEY_METRICS:
            cells_txt = []
            for c in present:
                r = cs[(cs.cell == c) & (cs.metric == key)]
                cells_txt.append(_fmt(r.iloc[0].to_dict()) if len(r) else "—")
            if any(t != "—" for t in cells_txt):
                L.append(f"| {label} | " + " | ".join(cells_txt) + " |")
        L.append("")

    if not ct.empty:
        L.append("## Paired contrasts (difference of cells within each seed; 95 % t-CI)\n")
        L.append("`*` = CI excludes 0. With 3 seeds the CI is wide; report it, don't over-read it.\n")
        for name, _, _ in CONTRASTS:
            sub = ct[ct.contrast == name]
            if sub.empty:
                continue
            L.append(f"### {name}\n")
            L.append("| Metric | mean Δ | 95 % CI | n | per-seed Δ |\n|---|---|---|---|---|")
            for _, r in sub.iterrows():
                ci = (f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]" if r.n > 1 else "—")
                L.append(f"| {r.label} | {r['mean']:+.4f}{' *' if r.ci_excludes_0 else ''} | "
                         f"{ci} | {r.n} | {r.per_seed} |")
            L.append("")

    L.append("## Gap-matched analysis\n")
    if fit_rows:
        L.append("Linear fit of threshold-attack AUC on the clean loss gap over all checkpoints, "
                 "per training type, evaluated on the overlapping gap range. Positive "
                 "`fgsm − standard` = FGSM-AT leaks more at the SAME generalisation gap. "
                 "Descriptive only (checkpoints of one run are correlated).\n")
        for key, (fits, rows) in fit_rows.items():
            L.append(f"**{key}** — standard: slope {fits['standard'][0]:.3f}, "
                     f"n={fits['standard'][4]}; FGSM: slope {fits['fgsm'][0]:.3f}, n={fits['fgsm'][4]}\n")
            if rows:
                L.append("| loss gap | standard | FGSM-AT | FGSM − std |\n|---|---|---|---|")
                for r in rows:
                    L.append(f"| {r['loss_gap']:.4f} | {r['standard']:.4f} | {r['fgsm']:.4f} | "
                             f"{r['fgsm_minus_standard']:+.4f} |")
            else:
                L.append("_No overlapping gap range between the two training types._")
            L.append("")
        L.append("![gap-matched](gap_matched.png)\n")
    else:
        L.append("_Not enough checkpoints yet._\n")
    if not df.empty:
        L.append("![cells](cell_comparison.png)\n")

    L.append("## Reference: numbers reported in the submitted paper (1 seed)\n")
    L.append("| Metric | Baseline (≈A) | Adversarial (≈D) |\n|---|---|---|")
    for m, a, d in PAPER_REFERENCE:
        L.append(f"| {m} | {a} | {d} |")
    L.append("\nThe original runs used one seed, trained the baseline with 30 max epochs / "
             "eta_min 0, augmented full-size PNGs, and drew their 5k eval members from a "
             "different (but also clean) sample of training images; differences to the new "
             "D − A contrast should be read with that in mind.\n")

    L.append("## Missing outputs\n")
    L.extend([f"- {m}" for m in missing] or ["- none"])
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=os.path.join(mc.MATCHED_DIR, "results"))
    args = p.parse_args()
    out_dir = os.path.join(args.results_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)

    df, points, missing = collect(args.results_dir)
    if df.empty:
        print("[AGG] no victims found yet.")
        return
    order = {c: i for i, c in enumerate(CELL_ORDER)}
    df = df.sort_values(["seed", "cell"], key=lambda s: s.map(order) if s.name == "cell" else s)
    df.to_csv(os.path.join(out_dir, "per_run.csv"), index=False)

    cs = cell_summary(df)
    cs.to_csv(os.path.join(out_dir, "cell_summary.csv"), index=False)
    ct = contrasts(df)
    ct.to_csv(os.path.join(out_dir, "contrasts.csv"), index=False)

    fit_rows = {}
    if not points.empty:
        points.to_csv(os.path.join(out_dir, "gap_matched_points.csv"), index=False)
        for key in ("loss_auc", "fluct_wb_auc", "adv_loss_auc"):
            r = gap_matched_fit(points, key)
            if r:
                fit_rows[key] = r
        plot_gap_matched(points, os.path.join(out_dir, "gap_matched.png"))
    plot_cell_comparison(df, os.path.join(out_dir, "cell_comparison.png"))
    write_report(args.results_dir, df, cs, ct, points, fit_rows, missing, out_dir)

    # Markdown copies of the two main tables
    with open(os.path.join(out_dir, "REPORT.md")) as f:
        report = f.read()
    for name, start, end in (("cell_summary.md", "## Per-cell", "## Paired"),
                             ("contrasts.md", "## Paired", "## Gap-matched"),
                             ("gap_matched_fit.md", "## Gap-matched", "## Reference")):
        if start in report:
            seg = report[report.index(start):report.index(end)] if end in report else report[report.index(start):]
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(seg)
    print(f"[AGG] {len(df)} victims, {len(points)} checkpoints → {out_dir}/REPORT.md")
    if missing:
        print(f"[AGG] {len(missing)} outputs still missing (listed in REPORT.md)")


if __name__ == "__main__":
    main()
