"""Signal extraction: query one model on every population record, clean and under
every probe, and keep a few scalars per (record, query).  All attacks are then
numpy-only, so every attack × access level × target is cheap.

Probe kinds (config PROBES[dataset] entries; ε in /255, σ in [0, 1] pixel units)
  hflip                       horizontal flip (benign query; LiRA averages it in)
  augset   shift=s            18 benign queries: {-s, 0, s}² translations × flip (LiRA-style)
  gauss    sigma=[..], k      k Gaussian-noise queries per σ (query-only "neighbourhood")
  uniform  eps=[..], k        k uniform-noise queries in the ε-ball (query-only)
  self_fgsm  eps=[..]         FGSM along the model's OWN input gradient (white-box)
  self_pgd   eps=[..], steps  PGD on the model itself (white-box)
  self_fgsm_own               as self_fgsm but w.r.t. the model's own predicted label
  tr_fgsm  src=F, eps=[..]    FGSM along the ensemble gradient of family F's source models
                              (transfer probe; grey/black-box)
  tr_fgsm_own src=F           same, labels = the queried model's own clean prediction
                              (label-free, as in the paper's AFMIA)
  tr_pgd   src=F, eps, steps  PGD on F's source ensemble, transferred
Probe names: "hflip", "augset", "gauss:s0.02", "self:fgsm:e2", "self:pgd10:e8",
"self_own:fgsm:e2", "tr:<F>:fgsm:e2", "tr_own:<F>:fgsm:e2", "tr:<F>:pgd5:e8".

Per single query q the store keeps  q.phi (LiRA logit-scaled true-class confidence),
q.ptrue, q.l1 (Σ|p_clean − p_q|, the paper's fluctuation), q.kl, q.flip (prediction
changed), q.corr (prediction correct), q.loss.  Multi-query probes keep the means of
those plus q.phi_std and q.phi_min.  "vec" probes also keep q.dp = |p_clean − p_q|
per class (the paper's AFMIA feature vector).  clean.logits keeps the full clean
output (Shokri's features).  self.gradnorm = ‖∇ₓ loss‖₂ when a self probe exists.
"""
import os
import time

import numpy as np
import torch

from . import adv
from .data import shift, to_float
from .train import channels_last, load_model, predict_logits
from .utils import log, stable_seed

STATS = ("phi", "ptrue", "l1", "kl", "flip", "corr", "loss")


def _fmt(v):
    return f"{v:g}"


def expand_probes(spec_list):
    """Config entries → flat list of probe dicts with unique names."""
    out = []
    for s in spec_list:
        k = s["kind"]
        vec = set(float(v) for v in s.get("vec", []))
        if k == "hflip":
            out.append(dict(name="hflip", kind=k, multi=False, vec=False))
        elif k == "augset":
            out.append(dict(name="augset", kind=k, multi=True, vec=False, shift=s.get("shift")))
        elif k == "gauss":
            for sg in s["sigma"]:
                out.append(dict(name=f"gauss:s{_fmt(sg)}", kind=k, sigma=float(sg), k=int(s.get("k", 8)),
                                multi=True, vec=False))
        elif k == "uniform":
            for e in s["eps"]:
                out.append(dict(name=f"uniform:e{_fmt(e)}", kind=k, eps=float(e), k=int(s.get("k", 8)),
                                multi=True, vec=False))
        elif k in ("self_fgsm", "self_fgsm_own"):
            pre = "self" if k == "self_fgsm" else "self_own"
            for e in s["eps"]:
                out.append(dict(name=f"{pre}:fgsm:e{_fmt(e)}", kind=k, eps=float(e), multi=False,
                                vec=float(e) in vec))
        elif k == "self_pgd":
            st = int(s.get("steps", 10))
            for e in s["eps"]:
                out.append(dict(name=f"self:pgd{st}:e{_fmt(e)}", kind=k, eps=float(e), steps=st,
                                multi=False, vec=float(e) in vec))
        elif k in ("tr_fgsm", "tr_fgsm_own"):
            pre = "tr" if k == "tr_fgsm" else "tr_own"
            for e in s["eps"]:
                out.append(dict(name=f"{pre}:{s['src']}:fgsm:e{_fmt(e)}", kind=k, src=s["src"],
                                eps=float(e), multi=False, vec=float(e) in vec))
        elif k == "tr_pgd":
            st = int(s.get("steps", 5))
            for e in s["eps"]:
                out.append(dict(name=f"tr:{s['src']}:pgd{st}:e{_fmt(e)}", kind=k, src=s["src"],
                                eps=float(e), steps=st, multi=False, vec=float(e) in vec))
        else:
            raise ValueError(f"unknown probe kind {k!r}")
    names = [p["name"] for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate probe names: {names}")
    return out


def probe_sources(probes):
    return sorted({p["src"] for p in probes if "src" in p})


def _stats(z0, zq, y, ml):
    p0, pq = adv.probs(z0, ml), adv.probs(zq, ml)
    return {"phi": adv.correctness_logodds(zq, y, ml),
            "ptrue": adv.true_class_prob(zq, y, ml),
            "l1": (p0 - pq).abs().sum(1),
            "kl": adv.kl_div(z0, zq, ml),
            "flip": ((z0 > 0) != (zq > 0)).float().mean(1) if ml else (z0.argmax(1) != zq.argmax(1)).float(),
            "corr": adv.correct(zq, y, ml),
            "loss": adv.per_example_loss(zq, y, ml)}, (p0 - pq).abs()


class _Store:
    def __init__(self, n):
        self.n, self.a = n, {}

    def put(self, key, sl, val, dtype=np.float32, width=None):
        if key not in self.a:
            shape = (self.n,) if width is None else (self.n, width)
            self.a[key] = np.zeros(shape, dtype)
        self.a[key][sl] = val.detach().float().cpu().numpy().astype(dtype)


def extract_signals(model_dir, ds, probes, source_dirs, out_path, device, bs, amp=True):
    """Query the model in `model_dir` on the whole population; write `out_path` (.npz)."""
    t0 = time.time()
    ml = ds.multilabel
    model = load_model(model_dir, ds, device)
    srcs = {f: [load_model(d, ds, device) for d in dirs] for f, dirs in source_dirs.items()}
    xs, ys = ds.x_pop.to(device), ds.y_pop.to(device)
    N = ds.n_pop
    st = _Store(N)
    s_px = max(1, round(2 * ds.img_size / 32))
    need_self_grad = any(p["kind"] == "self_fgsm" for p in probes)

    def fwd(m, x):
        with torch.no_grad(), adv.amp_ctx(device, amp):
            return m(channels_last(x)).float()

    for b, start in enumerate(range(0, N, bs)):
        sl = slice(start, min(start + bs, N))
        x = to_float(xs[sl])
        y = ys[sl]
        z0 = fwd(model, x)
        st.put("clean.logits", sl, z0, np.float16, z0.shape[1])
        st.put("clean.phi", sl, adv.correctness_logodds(z0, y, ml))
        st.put("clean.ptrue", sl, adv.true_class_prob(z0, y, ml))
        st.put("clean.loss", sl, adv.per_example_loss(z0, y, ml))
        st.put("clean.corr", sl, adv.correct(z0, y, ml))
        st.put("clean.mentr", sl, adv.modified_entropy(z0, y, ml))
        own = (z0 > 0).float() if ml else z0.argmax(1)
        g_self = adv.input_grad(model, x, y, ml, amp) if need_self_grad else None
        if g_self is not None:
            st.put("self.gradnorm", sl, g_self.flatten(1).norm(dim=1))
        g_self_own, tr_dir, tr_dir_own = None, {}, {}

        def record(p, zq):
            s, dp = _stats(z0, zq, y, ml)
            for k in STATS:
                st.put(f"{p['name']}.{k}", sl, s[k])
            if p.get("vec"):
                st.put(f"{p['name']}.dp", sl, dp, np.float16, dp.shape[1])

        def record_multi(p, zs):
            ss = [_stats(z0, zq, y, ml)[0] for zq in zs]
            for k in STATS:
                v = torch.stack([s[k] for s in ss])
                st.put(f"{p['name']}.{k}", sl, v.mean(0))
                if k == "phi":
                    st.put(f"{p['name']}.phi_std", sl, v.std(0))
                    st.put(f"{p['name']}.phi_min", sl, v.min(0).values)

        for p in probes:
            k = p["kind"]
            if k == "hflip":
                record(p, fwd(model, x.flip(3)))
            elif k == "augset":
                s = p.get("shift") or s_px
                zs = [fwd(model, (shift(x, dx, dy).flip(3) if fl else shift(x, dx, dy)))
                      for fl in (False, True) for dx in (-s, 0, s) for dy in (-s, 0, s)]
                record_multi(p, zs)
            elif k in ("gauss", "uniform"):
                zs = []
                for j in range(p["k"]):
                    gen = torch.Generator(device=x.device).manual_seed(stable_seed(p["name"], b, j))
                    if k == "gauss":
                        nz = torch.randn(x.shape, generator=gen, device=x.device) * p["sigma"]
                    else:
                        nz = (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * (p["eps"] / 255)
                    zs.append(fwd(model, (x + nz).clamp(0, 1)))
                record_multi(p, zs)
            elif k == "self_fgsm":
                record(p, fwd(model, adv.fgsm(x, g_self, p["eps"] / 255)))
            elif k == "self_fgsm_own":
                if g_self_own is None:
                    g_self_own = adv.input_grad(model, x, own, ml, amp)
                record(p, fwd(model, adv.fgsm(x, g_self_own, p["eps"] / 255)))
            elif k == "self_pgd":
                record(p, fwd(model, adv.pgd(model, x, y, ml, p["eps"] / 255, p["steps"], amp=amp)))
            elif k == "tr_fgsm":
                if p["src"] not in tr_dir:
                    tr_dir[p["src"]] = adv.ensemble_direction(srcs[p["src"]], x, y, ml, amp)
                record(p, fwd(model, adv.fgsm(x, tr_dir[p["src"]], p["eps"] / 255)))
            elif k == "tr_fgsm_own":
                if p["src"] not in tr_dir_own:
                    tr_dir_own[p["src"]] = adv.ensemble_direction(srcs[p["src"]], x, own, ml, amp)
                record(p, fwd(model, adv.fgsm(x, tr_dir_own[p["src"]], p["eps"] / 255)))
            elif k == "tr_pgd":
                xa = adv.pgd(srcs[p["src"]], x, y, ml, p["eps"] / 255, p["steps"], amp=amp)
                record(p, fwd(model, xa))
        if b % 20 == 0:
            log(f"    signals {sl.stop}/{N} ({(time.time() - t0) / 60:.1f} min)")

    arrays = dict(st.a)
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out_path)
    log(f"    signals written: {len(arrays)} arrays, {(time.time() - t0) / 60:.1f} min")


# ─── evasion and robust-overfitting diagnostics ───────────────────────────────

def evaluate_evasion(model_dir, ds, keep, eps_list, robust_eps, device, bs=500, n_gap=2000,
                     pgd_steps=10, amp=True):
    """Clean / FGSM / PGD performance on the test split, and the member vs non-member
    gap in clean and robust (PGD) correctness — the robust-overfitting signal."""
    from .train import task_metric
    ml = ds.multilabel
    model = load_model(model_dir, ds, device)
    out = {"metric_name": "mean AUC" if ml else "accuracy", "eps": list(eps_list), "test": {}}

    def adv_logits(x_u8, y_all, attack, eps):
        zs = []
        for i in range(0, len(x_u8), bs):
            x = to_float(x_u8[i:i + bs].to(device))
            y = y_all[i:i + bs].to(device)
            if attack == "fgsm":
                xa = adv.fgsm(x, adv.input_grad(model, x, y, ml, amp), eps / 255)
            else:
                xa = adv.pgd(model, x, y, ml, eps / 255, pgd_steps, random_start=True, amp=amp)
            with torch.no_grad(), adv.amp_ctx(device, amp):
                zs.append(model(channels_last(xa)).float().cpu())
        return torch.cat(zs)

    z_clean = predict_logits(model, ds.x_test, device, bs)
    out["test"]["clean"] = task_metric(z_clean, ds.y_test, ml)
    for e in eps_list:
        for a in ("fgsm", f"pgd{pgd_steps}"):
            z = adv_logits(ds.x_test, ds.y_test, "fgsm" if a == "fgsm" else "pgd", e)
            r = {"metric": task_metric(z, ds.y_test, ml),
                 "flip_rate": float((((z > 0) != (z_clean > 0)).float().mean()) if ml
                                    else (z.argmax(1) != z_clean.argmax(1)).float().mean())}
            out["test"][f"{a}_e{_fmt(e)}"] = r

    keep = np.asarray(keep, bool)
    rng = np.random.default_rng(0)
    mem = rng.choice(np.flatnonzero(keep), min(n_gap, keep.sum()), replace=False)
    non = rng.choice(np.flatnonzero(~keep), min(n_gap, (~keep).sum()), replace=False)
    gap = {"robust_eps": robust_eps}
    for name, ids in (("members", mem), ("nonmembers", non)):
        t = torch.from_numpy(np.sort(ids))
        xu, yy = ds.x_pop[t], ds.y_pop[t]
        zc = predict_logits(model, xu, device, bs)
        za = adv_logits(xu, yy, "pgd", robust_eps)
        gap[name] = {"clean_corr": float(adv.correct(zc, yy, ml).mean()),
                     "robust_corr": float(adv.correct(za, yy, ml).mean()),
                     "clean_loss": float(adv.per_example_loss(zc, yy, ml).mean()),
                     "robust_loss": float(adv.per_example_loss(za, yy, ml).mean())}
    gap["clean_gap"] = gap["members"]["clean_corr"] - gap["nonmembers"]["clean_corr"]
    gap["robust_gap"] = gap["members"]["robust_corr"] - gap["nonmembers"]["robust_corr"]
    out["membership_gap"] = gap
    return out
