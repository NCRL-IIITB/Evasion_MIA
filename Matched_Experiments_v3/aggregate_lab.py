"""
aggregate_lab.py
================
Report for the v3 experiment lab.  Builds everything the v2 report has
(paper-format Tables I–III and Fig. 6 / 7 numbers, the C-vs-D headline, the
paired bootstrap, evasion detail, gap-matched analysis) with the lab attacks
added, and on top of it:

  * LEADERBOARD — every attack (paper, v2, lab; Gradient Boosting and the
    classifier-free scores) ranked for each standard/adversarial pair of the
    same seed and regime:  by the trend (Δ AUC = D − C, paired-bootstrap CI)
    and by strength on the adversarial victim (AUC on D).
  * PATIENT-LEVEL metrics — the NIH split is patient-disjoint, and privacy is
    about patients: the membership score of a patient is the mean score of
    their evaluation images (AUC, TPR @ 1 % FPR, paired bootstrap over
    patients).  Computed from the saved per-image scores; no GPU needed.

Usage
-----
  python Matched_Experiments_v3/aggregate_lab.py --results_dir <results_v3 folder>
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(os.path.dirname(HERE), "Matched_Experiments_v2")
sys.path.insert(0, V2_DIR)
import v2_common as v2              # noqa: E402


def _load_v2(name):
    """Import a v2 module by file path — v1's Matched_Experiments/ has files with the
    same names (e.g. aggregate_results.py) and is also on sys.path."""
    spec = importlib.util.spec_from_file_location(f"v2_{name}", os.path.join(V2_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agg = _load_v2("aggregate_results")  # the v2 report

agg.ATTACK_DIRS = ["shadow_mia_std", "afmia", "shadow_mia_at", "lab"]
LAB_HEADLINE = [("Robustness-cal (perturbation only)", "Lab: robustness, reference-calibrated"),
                ("Query robustness (Square + noise)", "Lab: query-based robustness (no gradients)"),
                ("LiRA-cal confidence", "Lab: LiRA-style calibrated confidence"),
                ("All black-box (combined)", "Lab: all black-box features")]
agg.HEADLINE_MIA = agg.HEADLINE_MIA + LAB_HEADLINE

PAPER_SETS = {"Shadow-MIA", "Shadow-MIA + BCVA", "AFMIA (paper features)", "AFMIA (paper features) + BCVA"}
V2_SETS = {"AFMIA-v2 (perturbation only)", "AFMIA-v2 (+ clean confidence)", "Shadow-MIA (AT shadows)"}


def family(source, key):
    if source == "white-box":
        return "white-box (upper bound)"
    if key.startswith("threshold|"):
        return "black-box, no classifier"
    fs = key.split("|")[0]
    if fs in PAPER_SETS:
        return "paper"
    if fs in V2_SETS:
        return "v2"
    return "lab"


def patient_ids(R, seed):
    sp = v2.read_json(os.path.join(R, "splits", f"attack_split_seed{seed}.json"))
    if not sp:
        return None
    return np.array([n.split("_")[0] for n in sp["eval_members"] + sp["eval_nonmembers"]])


def to_patients(y, s, pid):
    df = pd.DataFrame({"p": pid, "y": y, "s": np.asarray(s, np.float64)})
    g = df.groupby("p", sort=True).agg(y=("y", "first"), s=("s", "mean"), n=("s", "size"))
    return g.y.values.astype(int), g.s.values, g.n.values


def score_sources(r):
    out = {}
    if r["scores"]:
        y = r["scores"].get("__y__")
        out.update({("attack", k): (y, v) for k, v in r["scores"].items() if k != "__y__"})
    ts = r["threshold_scores"]
    if ts is not None:
        out.update({("white-box", k): (ts["y_true"], v) for k, v in ts.items() if k != "y_true"})
    return out


def leaderboard(R, runs, pts_all, B):
    rows = []
    for c, d in agg.pairs_of(runs):
        pid = patient_ids(R, c["seed"])
        sc, sd = score_sources(c), score_sources(d)
        sub = pts_all[(pts_all.C == c["run"]) & (pts_all.D == d["run"])] if not pts_all.empty else pts_all
        for (source, key), (y_c, s_c) in sc.items():
            if (source, key) not in sd:
                continue
            y_d, s_d = sd[(source, key)]
            if y_c is None or len(y_c) != len(y_d) or not np.array_equal(y_c, y_d):
                continue
            img = sub[(sub.source == source) & (sub.key == key)] if len(sub) else sub
            if len(img) == 0:
                continue
            img = img.iloc[0]
            fs, clf = (key.split("|") + [""])[:2]
            if fs == "threshold":                       # classifier-free score, not a classifier
                clf = ""
            row = {"seed": c["seed"], "C": c["run"], "D": d["run"], "regime": agg._regime(c),
                   "family": family(source, key), "attack": key, "classifier": clf,
                   "auc_C": img.auc_c, "auc_D": img.auc_d, "d_auc": img.d_auc,
                   "d_auc_lo": img.d_auc_lo, "d_auc_hi": img.d_auc_hi,
                   "tpr1_C": img.tpr_c, "tpr1_D": img.tpr_d, "d_tpr1": img.d_tpr}
            if pid is not None and len(pid) == len(y_c):
                yp, pc, n_img = to_patients(y_c, s_c, pid)
                _, pd_, _ = to_patients(y_d, s_d, pid)
                st = agg.paired_bootstrap(yp, pc, pd_, B, seed=1)
                row.update({"n_patients": int(len(yp)), "pat_auc_C": st["auc_c"],
                            "pat_auc_D": st["auc_d"], "pat_d_auc": st["d_auc"],
                            "pat_d_auc_lo": st["d_auc_lo"], "pat_d_auc_hi": st["d_auc_hi"],
                            "pat_tpr1_C": st["tpr_c"], "pat_tpr1_D": st["tpr_d"]})
            for name, r in (("C", c), ("D", d)):
                x = agg.find_result(r, fs, clf) if source == "attack" and clf else None
                if x:
                    row[f"acc_{name}"] = x["accuracy"]
                    row[f"acc_median_{name}"] = x["acc_median"]
                    row[f"f1_{name}"] = x["f1"]
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(v, d=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def lab_section(lb, B):
    L = ["## Lab leaderboard — which attack shows the trend, and which is strongest\n",
         "One row per attack (Gradient Boosting for the trained attacks, plus every "
         "classifier-free score), for each standard (C) / adversarial (D) pair of the same seed "
         "and regime. **Δ AUC = AUC(D) − AUC(C)** with a paired-bootstrap 95 % CI over the "
         f"evaluation images (B = {B}); **patient** columns use the mean score of each patient's "
         "evaluation images. Families: *paper* = the submitted paper's attacks, *v2* = fixed "
         "AFMIA, *lab* = new attacks, *black-box, no classifier* = fixed-direction scores, "
         "*white-box* = upper bounds with full model access (not attacks the paper's threat "
         "model allows).\n"]
    if lb.empty:
        L.append("_Needs a standard and an adversarial victim of the same seed and regime in this "
                 "folder — merge the accounts' outputs first._\n")
        return L
    show = lb[(lb.classifier.isin(["", agg.PRIMARY]))]
    for (c, d), sub in show.groupby(["C", "D"], sort=False):
        L.append(f"### `{c}` (C) vs `{d}` (D)\n")
        L.append("**Ranked by the trend (Δ AUC, D − C)** — positive with CI above 0 = the attack "
                 "leaks more on the adversarially trained model.\n")
        L.append("| # | Family | Attack | AUC C | AUC D | Δ AUC [95 % CI] | TPR@1% C | TPR@1% D | "
                 "Patient AUC C | Patient AUC D | Patient Δ [95 % CI] | Acc C | Acc D |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(sub.sort_values("d_auc", ascending=False).iterrows(), 1):
            pat = (f"{r.pat_d_auc:+.4f} [{r.pat_d_auc_lo:+.4f}, {r.pat_d_auc_hi:+.4f}]"
                   if "pat_d_auc" in r and pd.notna(r.get("pat_d_auc")) else "—")
            L.append(f"| {i} | {r.family} | {r.attack.replace('|', ' / ')} | {fmt(r.auc_C)} | "
                     f"{fmt(r.auc_D)} | "
                     f"{r.d_auc:+.4f} [{r.d_auc_lo:+.4f}, {r.d_auc_hi:+.4f}] | {fmt(r.tpr1_C)} | "
                     f"{fmt(r.tpr1_D)} | {fmt(r.get('pat_auc_C'))} | {fmt(r.get('pat_auc_D'))} | "
                     f"{pat} | {fmt(r.get('acc_C'))} | {fmt(r.get('acc_D'))} |")
        bb = sub[sub.family != "white-box (upper bound)"].sort_values("auc_D", ascending=False).head(10)
        L.append("\n**Strongest black-box attacks on the adversarial victim (AUC on D)**\n")
        L.append("| # | Family | Attack | AUC D | TPR@1% D | Patient AUC D | AUC C |\n|---|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(bb.iterrows(), 1):
            L.append(f"| {i} | {r.family} | {r.attack.replace('|', ' / ')} | {fmt(r.auc_D)} | "
                     f"{fmt(r.tpr1_D)} | "
                     f"{fmt(r.get('pat_auc_D'))} | {fmt(r.auc_C)} |")
        L.append("")
    L.append("All classifiers and every pair: `summary/leaderboard.csv`.\n")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    R = args.results_dir
    runs, pts_all = agg.run(R, args.bootstrap)
    if not runs:
        return
    lb = leaderboard(R, runs, pts_all, args.bootstrap)
    out_dir = os.path.join(R, "summary")
    if not lb.empty:
        lb.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)
    lab = "\n".join(lab_section(lb, args.bootstrap)) + "\n"
    rep_path = os.path.join(out_dir, "REPORT.md")
    with open(rep_path) as f:
        rep = f.read()
    rep = rep.replace("# Matched experiment v2 — results", "# Matched experiment v3 (lab) — results", 1)
    anchor = "## Victims"
    rep = rep.replace(anchor, lab + "\n" + anchor, 1) if anchor in rep else rep + "\n" + lab
    with open(rep_path, "w") as f:
        f.write(rep)
    with open(os.path.join(out_dir, "LEADERBOARD.md"), "w") as f:
        f.write(lab)
    print(f"[LAB] leaderboard: {len(lb)} rows → {rep_path}")


if __name__ == "__main__":
    main()
