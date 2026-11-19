"""Training of one zoo model on its half of the population — resumable per epoch.

Training methods (spec["train"]["method"]):
  std       standard ERM
  nfgsm     N-FGSM adversarial training (single step, no catastrophic overfitting)
  fgsm_rs   FGSM with random start (Wong et al. 2020)
  pgd       PGD-k adversarial training (Madry et al. 2018), random start
  trades    TRADES (Zhang et al. 2019), β · KL(p(x) || p(x')) with a k-step inner max
Optional: clean_weight w mixes the clean loss in: (1 - w)·L_adv + w·L_clean.

No early stopping and no held-out selection: the last epoch is kept, which is
what an overfit deployed model looks like and what LiRA/RMIA papers evaluate.
"""
import math
import os
import time

import numpy as np
import torch

from . import adv
from .data import random_crop_flip, to_float
from .models import build_model
from .utils import EXIT_PAUSED, load_json, log, save_json, stable_seed, time_left

DEFAULT_TRAIN = dict(method="std", eps=8.0, steps=5, beta=6.0, clean_weight=0.0, nfgsm_k=2.0,
                     epochs=50, batch=256, lr=0.1, wd=5e-4, momentum=0.9, opt="sgd",
                     aug=True, aug_flip=True, amp=True, pct_start=0.15, eval_every=10)


def train_params(spec):
    p = dict(DEFAULT_TRAIN)
    p.update(spec.get("train", {}))
    return p


def channels_last(x):
    return x.contiguous(memory_format=torch.channels_last) if x.is_cuda else x


@torch.no_grad()
def predict_logits(model, x_u8, device, bs=500, amp=True):
    """Logits (CPU float32) of uint8 images (CPU or GPU tensor)."""
    model.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        xb = channels_last(to_float(x_u8[i:i + bs].to(device, non_blocking=True)))
        with adv.amp_ctx(device, amp):
            out.append(model(xb).float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


def task_metric(z, y, multilabel):
    """Accuracy (multi-class) or mean per-label ROC AUC (multi-label)."""
    if not multilabel:
        return float((z.argmax(1) == y).float().mean())
    from sklearn.metrics import roc_auc_score
    aucs = []
    yn, zn = y.numpy(), z.numpy()
    for c in range(yn.shape[1]):
        if 0 < yn[:, c].sum() < len(yn):
            aucs.append(roc_auc_score(yn[:, c], zn[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def save_weights(model, path):
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)


def load_model(model_dir, ds, device):
    meta = load_json(os.path.join(model_dir, "meta.json"))
    m = build_model(meta["arch"], ds.num_classes, ds.in_ch, ds.img_size, ds.mean, ds.std)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    m = m.to(device)
    if torch.device(device).type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    return m.eval()


def train_model(spec, ds, out_dir, device):
    """Returns "done" or "paused".  spec: model_id, arch, seed, keep (bool array), train."""
    os.makedirs(out_dir, exist_ok=True)
    final, ckpt_path = os.path.join(out_dir, "model.pt"), os.path.join(out_dir, "ckpt.pt")
    if os.path.exists(final) and os.path.exists(os.path.join(out_dir, "meta.json")):
        return "done"
    p = train_params(spec)
    ml = ds.multilabel
    keep = np.asarray(spec["keep"], bool)
    idx = torch.from_numpy(np.flatnonzero(keep))
    xs = ds.x_pop[idx].to(device)
    ys = ds.y_pop[idx].to(device)
    n = len(idx)
    steps_per_epoch = math.ceil(n / p["batch"])
    total = steps_per_epoch * p["epochs"]

    torch.manual_seed(spec["seed"])
    model = build_model(spec["arch"], ds.num_classes, ds.in_ch, ds.img_size, ds.mean, ds.std).to(device)
    if torch.device(device).type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if p["opt"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=p["lr"], momentum=p["momentum"],
                              weight_decay=p["wd"], nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=total,
                                                pct_start=p["pct_start"], anneal_strategy="linear",
                                                div_factor=25.0, final_div_factor=1e3)
    use_amp = p["amp"] and torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, history = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, history = ck["epoch"], ck["history"]
        log(f"  resumed {spec['model_id']} at epoch {start_epoch}/{p['epochs']}")

    eps = p["eps"] / 255.0
    pad = max(1, ds.img_size // 8)
    t_start = time.time()
    for epoch in range(start_epoch, p["epochs"]):
        t0 = time.time()
        model.train()
        g = torch.Generator().manual_seed(stable_seed(spec["seed"], "perm", epoch))
        perm = torch.randperm(n, generator=g).to(device)
        torch.manual_seed(stable_seed(spec["seed"], "aug", epoch))
        tot_loss, tot_corr, seen = 0.0, 0.0, 0
        for b in range(steps_per_epoch):
            bi = perm[b * p["batch"]:(b + 1) * p["batch"]]
            x = to_float(xs[bi])
            y = ys[bi]
            if p["aug"]:
                x = random_crop_flip(x, pad, p["aug_flip"])
            x = channels_last(x)
            method = p["method"]
            if method == "std":
                x_in = x
            elif method == "nfgsm":
                x_in = adv.nfgsm_example(model, x, y, ml, eps, p["nfgsm_k"], use_amp)
            elif method == "fgsm_rs":
                x_in = adv.fgsm_rs_example(model, x, y, ml, eps, use_amp)
            elif method == "pgd":
                x_in = adv.pgd(model, x, y, ml, eps, p["steps"], random_start=True, amp=use_amp)
            elif method == "trades":
                x_in = adv.trades_example(model, x, ml, eps, p["steps"], use_amp)
            else:
                raise ValueError(f"unknown training method {method!r}")
            with adv.amp_ctx(device, use_amp):
                if method == "trades":
                    z_c = model(x)
                    z_a = model(x_in)
                    loss = adv.per_example_loss(z_c, y, ml).mean() + \
                        p["beta"] * adv.kl_div(z_c, z_a, ml).mean()
                    z_log = z_c
                else:
                    z_a = model(x_in)
                    loss = adv.per_example_loss(z_a, y, ml).mean()
                    z_log = z_a
                    if method != "std" and p["clean_weight"] > 0:
                        z_c = model(x)
                        loss = (1 - p["clean_weight"]) * loss + \
                            p["clean_weight"] * adv.per_example_loss(z_c, y, ml).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += float(loss.detach()) * len(bi)
            tot_corr += float(adv.correct(z_log.detach().float(), y, ml).sum())
            seen += len(bi)
        rec = {"epoch": epoch + 1, "loss": tot_loss / seen, "train_acc_running": tot_corr / seen,
               "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        if (epoch + 1) % p["eval_every"] == 0 or epoch + 1 == p["epochs"]:
            z = predict_logits(model, ds.x_test, device)
            rec["test_metric"] = task_metric(z, ds.y_test, ml)
            model.train()
        history.append(rec)
        log(f"  [{spec['model_id']}] epoch {epoch + 1}/{p['epochs']} loss {rec['loss']:.4f} "
            f"acc {rec['train_acc_running']:.4f}" +
            (f" test {rec['test_metric']:.4f}" if "test_metric" in rec else "") + f" ({rec['seconds']}s)")
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history}
        torch.save(ck, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        epoch_s = time.time() - t0
        if epoch + 1 < p["epochs"] and time_left() < 1.5 * epoch_s + 180:
            log(f"  [{spec['model_id']}] pausing before the session deadline")
            return "paused"

    # final metrics (eval mode)
    sub = idx[: min(n, 5000)]
    z_tr = predict_logits(model, ds.x_pop[sub], device)
    z_te = predict_logits(model, ds.x_test, device)
    train_m, test_m = task_metric(z_tr, ds.y_pop[sub], ml), task_metric(z_te, ds.y_test, ml)
    save_weights(model, final)
    meta = {"model_id": spec["model_id"], "family": spec["family"], "arch": spec["arch"],
            "dataset": ds.key, "multilabel": ds.multilabel, "seed": spec["seed"],
            "n_train": int(n), "train": p,
            "train_metric": train_m, "test_metric": test_m, "gap": train_m - test_m,
            "metric_name": "mean AUC" if ml else "accuracy", "history": history,
            "train_seconds_this_session": round(time.time() - t_start, 1)}
    save_json(os.path.join(out_dir, "meta.json"), meta)
    np.save(os.path.join(out_dir, "keep.npy"), keep)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    log(f"  [{spec['model_id']}] done: train {train_m:.4f} test {test_m:.4f}")
    return "done"


__all__ = ["train_model", "load_model", "predict_logits", "task_metric", "EXIT_PAUSED"]
