"""Run the attack suite for every (setting, target) whose signals exist.  Results
are cached per pair and recomputed only when the number of references changed."""
import os
import re
import time

import numpy as np

from . import __version__
from .attacks import Sig, allowed_probes, run_attacks
from .utils import load_json, log, save_json, stable_seed, time_left


def _eps(name):
    m = re.search(r":e([0-9.]+)$", name)
    return float(m.group(1)) if m else 0.0


def resolve_song_probe(setting, allowed):
    """Song et al.'s attack uses the strongest adversarial example the attacker can make."""
    if setting.get("song_probe"):
        return setting["song_probe"] if setting["song_probe"] in allowed else None
    order = [r"^self:pgd", r"^tr:.*:pgd", r"^self:fgsm", r"^tr:.*:fgsm"]
    for pat in order:
        c = [p for p in allowed if re.search(pat, p)]
        if c:
            return max(c, key=_eps)
    return None


def resolve_paper_probe(setting, allowed, target):
    """The paper's AFMIA probe: a transfer (or, white-box, own-gradient) FGSM with per-class |Δp|."""
    if setting.get("paper_probe"):
        p = setting["paper_probe"]
        return p if p in allowed and target.has(f"{p}.dp") else None
    for pat in (r"^self:fgsm", r"^tr_own:", r"^tr:.*:fgsm"):
        c = [p for p in allowed if re.search(pat, p) and target.has(f"{p}.dp")]
        if c:
            return min(c, key=_eps)
    return None


def run_all(plan, res_dir, min_left_s=600):
    sig_dir = os.path.join(res_dir, "signals")
    out_dir = os.path.join(res_dir, "attacks")
    os.makedirs(out_dir, exist_ok=True)
    cache = {}
    meta_cache = {}

    def ready(mid):
        return os.path.exists(os.path.join(sig_dir, f"{mid}.npz"))

    def sig(mid):
        if mid not in cache:
            keep = np.load(os.path.join(res_dir, "models", mid, "keep.npy"))
            cache[mid] = Sig(os.path.join(sig_dir, f"{mid}.npz"), keep, mid)
        return cache[mid]

    def multilabel(mid):
        if mid not in meta_cache:
            meta_cache[mid] = load_json(os.path.join(res_dir, "models", mid, "meta.json"), {})
        return bool(meta_cache[mid].get("multilabel", False))

    n_done = n_skip = 0
    for s in plan.settings:
        ds = plan.dataset_of_setting(s)
        names = [p["name"] for p in plan.probes.get(ds, [])]
        allowed = allowed_probes(names, s["sources"], s["white_box"])
        if s.get("probe_filter"):              # keep only probes matching one of the regexes
            keep_benign = [p for p in allowed if p in ("hflip", "augset")]
            allowed = sorted(set(keep_benign) | {p for p in allowed
                                                 if any(re.search(pat, p) for pat in s["probe_filter"])})
        for t in plan.targets_of(s):
            if not ready(t):
                n_skip += 1
                continue
            refs = [r for r in plan.refs_for(s, t) if ready(r)]
            path = os.path.join(out_dir, f"{s['name']}__{t}.json")
            prev = load_json(path)
            if prev and prev.get("n_refs") == len(refs) and prev.get("version") == __version__:
                n_done += 1
                continue
            if time_left() < min_left_s:
                log("  attacks: stopping before the session deadline")
                return "paused"
            t0 = time.time()
            ts = sig(t)
            song = resolve_song_probe(s, allowed)
            paper = resolve_paper_probe(s, allowed, ts)
            res, info = run_attacks(ts, [sig(r) for r in refs], allowed, s["white_box"], multilabel(t),
                                    seed=stable_seed(s["name"], t), song_probe=song, paper_probe=paper,
                                    detail=s.get("detail", True))
            save_json(path, {"setting": s["name"], "group": s.get("group", s["name"]),
                             "access": s.get("access"), "target": t, "target_family": s["target"],
                             "ref_families": s["refs"], "sources": s["sources"],
                             "white_box": s["white_box"], "refs": refs, "n_refs": len(refs),
                             "song_probe": song, "paper_probe": paper, "info": info,
                             "version": __version__, "seconds": round(time.time() - t0, 1),
                             "results": res})
            best = max((r for r in res if r["category"] != "detail"), key=lambda r: r["tpr@1%"])
            log(f"  attacks {s['name']} / {t}: {len(refs)} refs, {len(res)} rows, "
                f"best TPR@1% {best['tpr@1%']:.3f} ({best['attack']}) in {time.time() - t0:.0f}s")
            n_done += 1
    log(f"  attacks: {n_done} (setting, target) pairs done, {n_skip} waiting for signals")
    return "done"
