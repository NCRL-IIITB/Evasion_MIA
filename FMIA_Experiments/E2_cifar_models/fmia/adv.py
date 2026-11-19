"""Losses, input gradients and L∞ attacks shared by training, probing and evasion.

All images are in [0, 1]; ε and step sizes are in the same units (e.g. 8/255).
Multi-label tasks (ChestMNIST) use per-label sigmoid/BCE; multi-class tasks
softmax/CE.  Per-example loss = CE, or the mean BCE over labels.
"""
import contextlib

import torch
import torch.nn.functional as F


def amp_ctx(device, enabled=True):
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def per_example_loss(z, y, multilabel):
    z = z.float()
    if multilabel:
        return F.binary_cross_entropy_with_logits(z, y, reduction="none").mean(1)
    return F.cross_entropy(z, y, reduction="none")


def probs(z, multilabel):
    z = z.float()
    return torch.sigmoid(z) if multilabel else torch.softmax(z, 1)


def log_probs(z, multilabel):
    """Per-class log p and log(1-p) (multi-label) or log-softmax (multi-class)."""
    z = z.float()
    if multilabel:
        return F.logsigmoid(z), F.logsigmoid(-z)
    return torch.log_softmax(z, 1), None


def kl_div(z_ref, z_q, multilabel):
    """Per-example KL( p(z_ref) || p(z_q) ); mean over labels for multi-label."""
    if multilabel:
        lp, lq = F.logsigmoid(z_ref.float()), F.logsigmoid(z_q.float())
        lp1, lq1 = F.logsigmoid(-z_ref.float()), F.logsigmoid(-z_q.float())
        p = lp.exp()
        return (p * (lp - lq) + (1 - p) * (lp1 - lq1)).mean(1)
    lp, lq = torch.log_softmax(z_ref.float(), 1), torch.log_softmax(z_q.float(), 1)
    return (lp.exp() * (lp - lq)).sum(1)


def correctness_logodds(z, y, multilabel):
    """LiRA's logit-scaled confidence φ = log p_y − log(1 − p_y), computed stably.
    Multi-label: mean over labels of the signed logit (2y − 1)·z (the per-label log-odds
    of the correct label)."""
    z = z.float()
    if multilabel:
        return ((2 * y - 1) * z).mean(1)
    zy = z.gather(1, y[:, None]).squeeze(1)
    other = z.scatter(1, y[:, None], float("-inf"))
    return zy - torch.logsumexp(other, 1)


def true_class_prob(z, y, multilabel):
    """p(y|x): softmax prob of the label, or the geometric mean over labels of p(correct)."""
    z = z.float()
    if multilabel:
        return F.logsigmoid((2 * y - 1) * z).mean(1).exp()
    return torch.softmax(z, 1).gather(1, y[:, None]).squeeze(1)


def modified_entropy(z, y, multilabel, eps=1e-12):
    """Song & Mittal (USENIX Sec 2021): −(1−p_y) log p_y − Σ_{i≠y} p_i log(1−p_i).
    Multi-label: per label (binary), −2(1−q) log q with q = p(correct label); mean over labels."""
    z = z.float()
    if multilabel:
        q = torch.sigmoid((2 * y - 1) * z)
        return (-2 * (1 - q) * torch.log(q.clamp_min(eps))).mean(1)
    p = torch.softmax(z, 1)
    py = p.gather(1, y[:, None]).squeeze(1)
    other = -(p * torch.log((1 - p).clamp_min(eps))).sum(1) + py * torch.log((1 - py).clamp_min(eps))
    return -(1 - py) * torch.log(py.clamp_min(eps)) + other


def predict(z, multilabel):
    return (z > 0) if multilabel else z.argmax(1)


def correct(z, y, multilabel):
    if multilabel:
        return ((z > 0) == (y > 0.5)).float().mean(1)
    return (z.argmax(1) == y).float()


# ─── gradients ────────────────────────────────────────────────────────────────

def input_grad(model, x, y, multilabel, amp=True, loss_fn=None, scale=256.0):
    """d/dx Σ_i loss_i — per-example input gradients (fp32), with fp16 autocast and a
    loss scale against underflow; falls back to fp32 if the result is not finite."""
    loss_fn = loss_fn or (lambda z: per_example_loss(z, y, multilabel))
    xr = x.detach().clone().requires_grad_(True)
    with amp_ctx(x.device, amp):
        z = model(xr)
    g, = torch.autograd.grad(loss_fn(z).sum() * scale, xr)
    g = g.float() / scale
    if amp and not torch.isfinite(g).all():
        xr = x.detach().clone().requires_grad_(True)
        g, = torch.autograd.grad(loss_fn(model(xr)).sum(), xr)
        g = g.float()
    return g


def ensemble_direction(models, x, y, multilabel, amp=True):
    """Σ_k ĝ_k with ĝ_k the input gradient of model k normalised to unit mean |·| per example."""
    tot = torch.zeros_like(x)
    for m in models:
        g = input_grad(m, x, y, multilabel, amp)
        tot += g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-20)
    return tot


def fgsm(x, direction, eps):
    return (x + eps * direction.sign()).clamp_(0, 1)


def pgd(models, x, y, multilabel, eps, steps, alpha=None, random_start=False, amp=True,
        generator=None):
    """L∞ PGD maximising the (normalised) summed loss of `models` (one model = standard PGD)."""
    models = models if isinstance(models, (list, tuple)) else [models]
    alpha = alpha if alpha is not None else 2.5 * eps / max(steps, 1)
    xa = x.clone()
    if random_start:
        u = torch.rand(x.shape, device=x.device, generator=generator)
        xa = (x + (2 * u - 1) * eps).clamp_(0, 1)
    for _ in range(steps):
        d = ensemble_direction(models, xa, y, multilabel, amp) if len(models) > 1 else \
            input_grad(models[0], xa, y, multilabel, amp)
        xa = xa + alpha * d.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp_(0, 1)
    return xa.detach()


# ─── adversarial-training inner steps (model in train mode) ───────────────────

def nfgsm_example(model, x, y, multilabel, eps, k=2.0, amp=True):
    """N-FGSM (de Jorge et al., NeurIPS 2022): noise U(-kε, kε), one step of size ε, no
    projection.  Single-step AT without catastrophic overfitting up to large ε."""
    noise = (torch.rand_like(x) * 2 - 1) * (k * eps)
    xn = (x + noise).clamp(0, 1)
    g = input_grad(model, xn, y, multilabel, amp)
    return (xn + eps * g.sign()).clamp_(0, 1).detach()


def fgsm_rs_example(model, x, y, multilabel, eps, amp=True):
    """FGSM-RS (Wong et al., ICLR 2020): uniform start, step 1.25ε, projected."""
    d0 = (torch.rand_like(x) * 2 - 1) * eps
    xs = (x + d0).clamp(0, 1)
    g = input_grad(model, xs, y, multilabel, amp)
    xa = xs + 1.25 * eps * g.sign()
    return torch.min(torch.max(xa, x - eps), x + eps).clamp_(0, 1).detach()


def trades_example(model, x, multilabel, eps, steps, amp=True):
    """TRADES inner maximisation of KL(p(x) || p(x')) (Zhang et al., ICML 2019); eval mode."""
    was_training = model.training
    model.eval()
    with torch.no_grad(), amp_ctx(x.device, amp):
        z0 = model(x).float()
    xa = (x + 0.001 * torch.randn_like(x)).clamp(0, 1)
    alpha = 2.5 * eps / max(steps, 1)
    for _ in range(steps):
        g = input_grad(model, xa, None, multilabel, amp,
                       loss_fn=lambda z: kl_div(z0, z, multilabel))
        xa = xa + alpha * g.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp_(0, 1)
    model.train(was_training)
    return xa.detach()
