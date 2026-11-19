"""summary/REPORT.md, CSVs and figures from the cached attack results."""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from .attacks import EXISTING, LABELS, OURS
from .metrics import FPR_GRID
from .plan import describe_family
from .utils import load_json

ONLINE_BASE = ["lira_on", "lira_on_pv", "lira_aug_on", "rmia_on"]
OFFLINE_BASE = ["loss", "mentr", "shokri", "lira_off", "rmia_off", "song_adv", "labelonly", "gradnorm"]
ROC_ATTACKS = ["loss", "shokri", "lira_off", "lira_on", "rmia_off", "rmia_on", "song_adv",
               "fmia_paper", "cfmia_fl_on", "cfmia_off", "cfmia_on"]
TRADEOFF_ATTACKS = ["shokri", "lira_on", "rmia_on", "song_adv", "fmia_paper", "cfmia_on"]
METRICS = [("auc", "AUC"), ("tpr@1%", "TPR@1%"), ("tpr@0.1%", "TPR@0.1%"), ("acc", "Acc")]


def load_results(res_dir):
    raw = [load_json(f) for f in sorted(glob.glob(os.path.join(res_dir, "attacks", "*.json")))]
    rows = []
    for j in raw:
        for r in j["results"]:
            rows.append({"setting": j["setting"], "group": j.get("group", j["setting"]),
                         "access": j.get("access"), "target": j["target"],
                         "target_family": j.get("target_family"), "n_refs": j["n_refs"],
                         "attack": r["attack"], "label": r["label"], "category": r["category"],
                         "offline": r.get("offline"), "auc": r["auc"], "tpr@1%": r["tpr@1%"],
                         "tpr@0.1%": r["tpr@0.1%"], "acc": r["acc"], "roc": r.get("roc_tpr"),
                         "column": r.get("column") or r.get("probe")})
    return raw, rows


def aggregate(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["setting"], r["attack"])].append(r)
    agg = {}
    for (s, a), rs in g.items():
        d = {"setting": s, "attack": a, "label": rs[0]["label"], "category": rs[0]["category"],
             "n": len(rs), "n_refs": int(np.mean([r["n_refs"] for r in rs])), "group": rs[0]["group"],
             "columns": sorted({str(r["column"]) for r in rs if r["column"]})}
        for k, _ in METRICS:
            v = np.array([r[k] for r in rs], float)
            d[k], d[k + "_std"] = float(np.nanmean(v)), float(np.nanstd(v)) if len(v) > 1 else 0.0
        rocs = [r["roc"] for r in rs if r["roc"]]
        d["roc"] = np.mean(rocs, 0).tolist() if rocs else None
        agg[(s, a)] = d
    return agg


def _f(d, k, pct=False):
    m, s = d[k], d[k + "_std"]
    if pct:
        return f"{100 * m:.1f}" + (f" ± {100 * s:.1f}" if d["n"] > 1 else "")
    return f"{m:.3f}" + (f" ± {s:.3f}" if d["n"] > 1 else "")


def _best(agg, setting, attacks, key="tpr@1%"):
    c = [agg[(setting, a)] for a in attacks if (setting, a) in agg]
    return max(c, key=lambda d: (d[key], d["auc"])) if c else None


def _evasion_by_family(plan, res_dir):
    out = {}
    for fam in plan.families:
        evs, metas = [], []
        for m in plan.models.values():
            if m["family"] != fam:
                continue
            e = load_json(os.path.join(res_dir, "evasion", f"{m['model_id']}.json"))
            mt = load_json(os.path.join(res_dir, "models", m["model_id"], "meta.json"))
            if e:
                evs.append(e)
            if mt:
                metas.append(mt)
        out[fam] = (evs, metas)
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def robust_metric(evs):
    """Mean test metric under the strongest PGD attack in the evasion grid."""
    if not evs:
        return float("nan")
    keys = [k for k in evs[0]["test"] if k.startswith("pgd")]
    if not keys:
        return float("nan")
    k = max(keys, key=lambda s: float(s.split("_e")[1]))
    return _mean([e["test"][k]["metric"] for e in evs if k in e["test"]])


def write_report(plan, res_dir, status_line=""):
    out_dir = os.path.join(res_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)
    raw, rows = load_results(res_dir)
    agg = aggregate(rows)
    with open(os.path.join(out_dir, "attack_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "access", "target", "n_refs", "attack", "category", "auc", "tpr@1%",
                    "tpr@0.1%", "acc", "column"])
        for r in rows:
            w.writerow([r["setting"], r["access"], r["target"], r["n_refs"], r["attack"], r["category"],
                        f"{r['auc']:.5f}", f"{r['tpr@1%']:.5f}", f"{r['tpr@0.1%']:.5f}", f"{r['acc']:.5f}",
                        r["column"] or ""])
    with open(os.path.join(out_dir, "leaderboard.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "attack", "label", "category", "n_targets", "n_refs"] +
                   [x for k, _ in METRICS for x in (k, k + "_std")])
        for d in agg.values():
            w.writerow([d["setting"], d["attack"], d["label"], d["category"], d["n"], d["n_refs"]] +
                       [f"{d[x]:.5f}" for k, _ in METRICS for x in (k, k + "_std")])

    cfg = plan.cfg
    L = [f"# {plan.experiment} — results" + (" (SMOKE TEST)" if plan.smoke else ""), ""]
    L += [getattr(cfg, "DESCRIPTION", "").strip(), ""]
    if status_line:
        L += [f"**Status:** {status_line}", ""]

    # 1. models
    ev = _evasion_by_family(plan, res_dir)
    L += ["## 1. Victim / reference models", "",
          "Train and test metric: accuracy, or mean per-label AUC for multi-label data. "
          "*Robust* columns: PGD-10 at the dataset's robust ε on 2k members vs 2k non-members "
          "(fraction correct); the member − non-member robust gap is the robust-overfitting "
          "signal that fluctuation attacks exploit.", "",
          "| Family | Dataset | Model | Trained | Train | Test | Gap | Test under strongest PGD | "
          "Clean corr. M − NM | Robust corr. M − NM |", "|---|---|---|---|---|---|---|---|---|---|"]
    for fam, (evs, metas) in ev.items():
        f = plan.families[fam]
        n_tot = f["n_models"]
        gaps = [e["membership_gap"] for e in evs]
        L.append(f"| `{fam}` | {f['dataset']} | {describe_family(plan, fam)} | {len(metas)}/{n_tot} | "
                 f"{_mean([m['train_metric'] for m in metas]):.3f} | {_mean([m['test_metric'] for m in metas]):.3f} | "
                 f"{_mean([m['gap'] for m in metas]):+.3f} | {robust_metric(evs):.3f} | "
                 f"{_mean([g['clean_gap'] for g in gaps]):+.3f} | {_mean([g['robust_gap'] for g in gaps]):+.3f} |")
    L += [""]
    any_ev = next((evs for evs, _ in ev.values() if evs), None)
    if any_ev:
        keys = list(any_ev[0]["test"].keys())
        L += ["**Evasion (test split, mean over the family's models):** metric / flip rate", "",
              "| Family | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for fam, (evs, _) in ev.items():
            if not evs:
                continue
            cells = []
            for k in keys:
                if k == "clean":
                    cells.append(f"{_mean([e['test']['clean'] for e in evs]):.3f}")
                else:
                    cells.append(f"{_mean([e['test'][k]['metric'] for e in evs if k in e['test']]):.3f} / "
                                 f"{_mean([e['test'][k]['flip_rate'] for e in evs if k in e['test']]):.2f}")
            L.append(f"| `{fam}` | " + " | ".join(cells) + " |")
        L += [""]

    # 2. headline
    settings = [s for s in plan.settings if any((s["name"], a) in agg for a in EXISTING + OURS)]
    L += ["## 2. Headline — proposed CF-MIA vs the strongest existing attack", "",
          "TPR@1% FPR (in %) and AUC, mean over target models.  *Strongest existing* is the best "
          "baseline **on the target's labels** (max over the baselines of that regime), which favours "
          "the baselines; CF-MIA is the single pre-declared method.  Online = reference models that "
          "trained on the candidate exist (LiRA/RMIA online); offline = only OUT references.", "",
          "| Setting | Access | #targets | #refs | Best existing online | CF-MIA online | Δ TPR@1% | "
          "Best existing offline | CF-MIA offline | Δ TPR@1% | Paper fluct. MIA | Original MIA |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(d):
        return "—" if d is None else f"{100 * d['tpr@1%']:.1f} / {d['auc']:.3f}"

    def cellb(d):
        return "—" if d is None else f"{100 * d['tpr@1%']:.1f} / {d['auc']:.3f} ({d['attack']})"

    def delta(a, b):
        return "—" if (a is None or b is None) else f"{100 * (a['tpr@1%'] - b['tpr@1%']):+.1f}"

    for s in settings:
        n = s["name"]
        bo, bf = _best(agg, n, ONLINE_BASE), _best(agg, n, OFFLINE_BASE)
        co, cf = agg.get((n, "cfmia_on")), agg.get((n, "cfmia_off"))
        any_d = next((agg[(n, a)] for a in EXISTING + OURS if (n, a) in agg), None)
        L.append(f"| `{n}` | {s.get('access', '')} | {any_d['n'] if any_d else 0} | "
                 f"{any_d['n_refs'] if any_d else 0} | {cellb(bo)} | {cell(co)} | {delta(co, bo)} | "
                 f"{cellb(bf)} | {cell(cf)} | {delta(cf, bf)} | {cell(agg.get((n, 'fmia_paper')))} | "
                 f"{cell(agg.get((n, 'shokri')))} |")
    L += [""]

    # 3. per setting
    L += ["## 3. All attacks per setting", "",
          "Mean ± std over target models.  TPR in %.  ★ = highest TPR@1% in the setting.", ""]
    for s in settings:
        n = s["name"]
        L += [f"### `{n}`", ""]
        desc = [f"access **{s.get('access', '')}**", f"target family `{s['target']}`",
                f"references {', '.join('`' + r + '`' for r in s['refs'])}",
                f"transfer sources {', '.join('`' + r + '`' for r in s['sources']) or 'none'}",
                f"white-box probes {'yes' if s['white_box'] else 'no'}"]
        L += ["; ".join(desc) + "."]
        if s.get("note"):
            L += ["", s["note"]]
        L += ["", "| Attack | Type | AUC | TPR@1% | TPR@0.1% | Acc | Chosen probe / column |",
              "|---|---|---|---|---|---|---|"]
        ds = [agg[(n, a)] for a in EXISTING + OURS if (n, a) in agg]
        top = max(ds, key=lambda d: d["tpr@1%"]) if ds else None
        for d in ds:
            star = " ★" if d is top else ""
            L.append(f"| {d['label']}{star} | {'existing' if d['category'] == 'existing' else '**ours**'} | "
                     f"{_f(d, 'auc')} | {_f(d, 'tpr@1%', True)} | {_f(d, 'tpr@0.1%', True)} | {_f(d, 'acc')} | "
                     f"{', '.join(d['columns'][:2])} |")
        det = sorted([d for (sn, a), d in agg.items() if sn == n and d["category"] == "detail"],
                     key=lambda d: -d["auc"])
        if det:
            L += ["", "<details><summary>Single fluctuation statistics, ranked by AUC (top 12)</summary>", "",
                  "| Statistic | AUC | TPR@1% |", "|---|---|---|"]
            for d in det[:12]:
                L.append(f"| {d['label']} | {_f(d, 'auc')} | {_f(d, 'tpr@1%', True)} |")
            L += ["", "</details>"]
        if _plot_roc(agg, n, os.path.join(out_dir, f"roc_{_safe(n)}.png")):
            L += ["", f"![ROC {n}](roc_{_safe(n)}.png)"]
        L += [""]

    # 4. experiment-specific figures
    rep = getattr(cfg, "REPORT", {})
    if rep.get("tradeoff_settings"):
        p = os.path.join(out_dir, "tradeoff.png")
        if _plot_tradeoff(plan, agg, ev, rep["tradeoff_settings"], p, L):
            L += ["", "![trade-off](tradeoff.png)", ""]
    groups = sorted({s["group"] for s in plan.settings if s.get("group")})
    if groups:
        p = os.path.join(out_dir, "n_refs.png")
        if _plot_refs(plan, agg, groups, p, L):
            L += ["", "![number of reference models](n_refs.png)", ""]
    L += ["## Notes", "",
          "* Every learned or selected component of an attack (classifier, feature weights, probe, "
          "sign, RMIA's *a*) is fitted on reference models only, each reference playing pseudo-target "
          "against the others.  The target's membership labels are used only to score.",
          "* `Acc` is the best balanced accuracy over thresholds (as in LiRA/RMIA papers); it is "
          "optimistic for every attack alike.",
          "* Raw per-pair results: `attacks/<setting>__<target>.json`; flat table: "
          "`summary/attack_results.csv`; aggregated: `summary/leaderboard.csv`.", ""]
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    return os.path.join(out_dir, "REPORT.md")


def _safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_roc(agg, setting, path):
    curves = [(a, agg[(setting, a)]) for a in ROC_ATTACKS if (setting, a) in agg and agg[(setting, a)]["roc"]]
    if not curves:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for a, d in curves:
        ls = "-" if a.startswith(("cfmia", "fmia")) else "--"
        lw = 2.2 if a in ("cfmia_on", "cfmia_off") else 1.2
        ax.plot(FPR_GRID, np.maximum(d["roc"], 1e-4), ls, lw=lw, label=f"{a} ({100 * d['tpr@1%']:.1f}%)")
    ax.plot(FPR_GRID, FPR_GRID, ":", color="grey", lw=1)
    ax.set(xscale="log", yscale="log", xlim=(1e-4, 1), ylim=(1e-4, 1), xlabel="FPR", ylabel="TPR",
           title=setting)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _plot_tradeoff(plan, agg, ev, settings, path, L):
    pts = []
    for sn in settings:
        s = next((x for x in plan.settings if x["name"] == sn), None)
        if s is None:
            continue
        evs, _ = ev.get(s["target"], ([], []))
        pts.append((sn, robust_metric(evs), {a: agg.get((sn, a)) for a in TRADEOFF_ATTACKS}))
    pts = [p for p in pts if np.isfinite(p[1]) and any(p[2].values())]
    if len(pts) < 2:
        return False
    pts.sort(key=lambda p: p[1])
    L += ["## 4. Robustness–privacy trade-off", "",
          "x: test metric under the strongest PGD attack (higher = more robust, i.e. evasion works less); "
          "y: membership TPR@1% FPR.", "",
          "| Setting | Robust test metric | " + " | ".join(TRADEOFF_ATTACKS) + " |",
          "|---|---|" + "---|" * len(TRADEOFF_ATTACKS)]
    for sn, rx, ds in pts:
        L.append(f"| `{sn}` | {rx:.3f} | " +
                 " | ".join("—" if ds[a] is None else f"{100 * ds[a]['tpr@1%']:.1f}" for a in TRADEOFF_ATTACKS) + " |")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    xs = [p[1] for p in pts]
    for a in TRADEOFF_ATTACKS:
        ys = [p[2][a]["tpr@1%"] * 100 if p[2][a] else np.nan for p in pts]
        ax.plot(xs, ys, "o-" if a.startswith(("cfmia", "fmia")) else "s--", label=a,
                lw=2.2 if a == "cfmia_on" else 1.2)
    for sn, rx, _ in pts:
        ax.annotate(sn, (rx, 0), fontsize=6, rotation=60, xytext=(0, 4), textcoords="offset points")
    ax.set(xlabel="test metric under PGD (robustness)", ylabel="MIA TPR@1% FPR (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _plot_refs(plan, agg, groups, path, L):
    atts = ["lira_on", "lira_off", "rmia_on", "rmia_off", "cfmia_on", "cfmia_off"]
    plt = _plt()
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    L += ["## 5. Number of reference models", ""]
    drew = False
    for ax, g in zip(axes[0], groups):
        ss = sorted([s for s in plan.settings if s.get("group") == g], key=lambda s: s["n_refs"])
        L += [f"**`{g}`** — TPR@1% (%)", "", "| #refs | " + " | ".join(atts) + " |", "|---|" + "---|" * len(atts)]
        for s in ss:
            L.append(f"| {s['n_refs']} | " + " | ".join(
                "—" if (s["name"], a) not in agg else f"{100 * agg[(s['name'], a)]['tpr@1%']:.1f}" for a in atts) + " |")
        L += [""]
        for a in atts:
            xs = [s["n_refs"] for s in ss if (s["name"], a) in agg]
            ys = [100 * agg[(s["name"], a)]["tpr@1%"] for s in ss if (s["name"], a) in agg]
            if xs:
                drew = True
                ax.plot(xs, ys, "o-" if a.startswith("cfmia") else "s--", label=a)
        ax.set(xscale="log", xlabel="# reference models", ylabel="TPR@1% FPR (%)", title=g)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return drew
