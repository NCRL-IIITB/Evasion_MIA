"""Turn an experiment's config.py into concrete models, probes and attack settings.

config.py defines
  EXPERIMENT   folder name, used in seeds and the report title
  DATASETS     {key: dict(n_pop, n_test, extract_bs, evasion_eps, robust_eps)}
  TRAIN        default training hyper-parameters (see fmia/train.py DEFAULT_TRAIN)
  FAMILIES     [dict(name, dataset, arch (str | list, cycled), train={...}, n_models, n_sources)]
               models are  <family>_00 ... ; the first n_sources of a family are its
               SOURCE models (surrogates whose gradients build transfer probes) and are
               never used as targets or references.
  PROBES       {dataset key: [probe specs]}   (fmia/signals.py)
  SETTINGS     [dict(name, access, target, refs=[families], sources=[families], white_box,
                     n_targets, n_refs (int | list), song_probe, paper_probe, note)]
  SMOKE        overrides for a quick pipeline test
"""
import copy
import os

import numpy as np

from .signals import expand_probes, probe_sources
from .train import train_params
from .utils import stable_seed


class Plan:
    def __init__(self, cfg, smoke=False, dataset_override=None):
        self.cfg = cfg
        self.experiment = cfg.EXPERIMENT
        self.smoke = smoke
        sm = getattr(cfg, "SMOKE", {}) if smoke else {}
        self.datasets = copy.deepcopy(cfg.DATASETS)
        fams = copy.deepcopy(cfg.FAMILIES)
        probes = copy.deepcopy(cfg.PROBES)
        if dataset_override:                       # local tests: every dataset -> synthetic
            mapping = {k: dataset_override for k in self.datasets}
            self.datasets = {dataset_override: dict(next(iter(self.datasets.values())))}
            for f in fams:
                f["dataset"] = mapping[f["dataset"]]
            probes = {dataset_override: probes[next(iter(probes))]}
        if smoke:
            for d in self.datasets.values():
                d["n_pop"] = sm.get("n_pop", 1500)
                d["n_test"] = sm.get("n_test", 500)
            for f in fams:
                f["n_sources"] = min(f.get("n_sources", 0), 1)
                f["n_models"] = min(f["n_models"], f["n_sources"] + sm.get("max_models", 5))
                f.setdefault("train", {})["epochs"] = sm.get("epochs", 2)
        self.families = {f["name"]: f for f in fams}
        self.probes = {k: expand_probes(v) for k, v in probes.items()}
        self.train_default = dict(getattr(cfg, "TRAIN", {}))
        if smoke:
            self.train_default["epochs"] = sm.get("epochs", 2)
        self.settings = self._expand_settings(copy.deepcopy(cfg.SETTINGS), sm)
        self.models = self._models()

    # ---- models
    def _models(self):
        out = {}
        override = os.environ.get("FMIA_ARCH_OVERRIDE")     # local CPU tests only
        for fam in self.families.values():
            archs = fam["arch"] if isinstance(fam["arch"], list) else [fam["arch"]]
            if override:
                archs = [override]
            for i in range(fam["n_models"]):
                mid = f"{fam['name']}_{i:02d}"
                tr = dict(self.train_default)
                tr.update(fam.get("train", {}))
                tr = train_params({"train": tr})          # fill every default (method, eps, ...)
                out[mid] = dict(model_id=mid, family=fam["name"], index=i, dataset=fam["dataset"],
                                arch=archs[i % len(archs)], is_source=i < fam.get("n_sources", 0),
                                seed=stable_seed(self.experiment, mid), train=tr)
        return out

    def keep_mask(self, model_id, n_pop):
        m = self.models[model_id]
        rng = np.random.default_rng(stable_seed("keep", m["dataset"], model_id))
        return rng.random(n_pop) < 0.5

    def sources_of(self, family):
        return [m for m in self.models.values() if m["family"] == family and m["is_source"]]

    def nonsource(self, family):
        return [m for m in self.models.values() if m["family"] == family and not m["is_source"]]

    def probe_source_families(self, dataset):
        return probe_sources(self.probes.get(dataset, []))

    def signal_deps(self, model_id):
        """Models whose weights are needed to extract this model's signals."""
        ds = self.models[model_id]["dataset"]
        deps = [model_id]
        for f in self.probe_source_families(ds):
            deps += [m["model_id"] for m in self.sources_of(f)]
        return deps

    def train_order(self):
        """Sources first, then round-robin over families so a partial run covers every family."""
        srcs = [m for m in self.models.values() if m["is_source"]]
        prio = {f: self.families[f].get("priority", 0) for f in self.families}
        rest = sorted([m for m in self.models.values() if not m["is_source"]],
                      key=lambda m: (prio[m["family"]], m["index"], list(self.families).index(m["family"])))
        return [m["model_id"] for m in srcs + rest]

    # ---- settings
    def _expand_settings(self, settings, sm):
        out = []
        for s in settings:
            s.setdefault("refs", [s["target"]])
            s.setdefault("sources", [])
            s.setdefault("white_box", s.get("access") == "white")
            s.setdefault("n_targets", 4)
            if self.smoke:
                s["n_targets"] = 1
            nr = s.get("n_refs")
            if isinstance(nr, list):
                for k in nr:
                    t = dict(s)
                    t["n_refs"] = int(k)
                    t["name"] = f"{s['name']}@{k}refs"
                    t["group"] = s["name"]
                    out.append(t)
            else:
                out.append(s)
        names = [s["name"] for s in out]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate setting names: {names}")
        return out

    def targets_of(self, setting):
        return [m["model_id"] for m in self.nonsource(setting["target"])][: setting["n_targets"]]

    def refs_for(self, setting, target_id):
        refs = []
        for f in setting["refs"]:
            refs += [m["model_id"] for m in self.nonsource(f) if m["model_id"] != target_id]
        if setting.get("n_refs"):
            refs = refs[: setting["n_refs"]]
        return refs

    def dataset_of_setting(self, setting):
        return self.families[setting["target"]]["dataset"]


def describe_family(plan, fam_name):
    f = plan.families[fam_name]
    tr = dict(plan.train_default)
    tr.update(f.get("train", {}))
    p = train_params({"train": tr})
    if p["method"] == "std":
        how = "standard"
    elif p["method"] == "trades":
        how = f"TRADES ε={p['eps']:g}/255, {p['steps']} steps, β={p['beta']:g}"
    elif p["method"] == "pgd":
        how = f"PGD-{p['steps']} AT ε={p['eps']:g}/255"
    else:
        how = f"{p['method'].upper()} AT ε={p['eps']:g}/255"
    archs = []
    for m in plan.models.values():                 # the architectures actually trained
        if m["family"] == fam_name and m["arch"] not in archs:
            archs.append(m["arch"])
    arch = "/".join(archs)
    return f"{arch}, {how}, {p['epochs']} ep" + ("" if p["aug"] else ", no aug")
