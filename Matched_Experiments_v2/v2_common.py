"""
v2_common.py
============
Shared building blocks for Matched_Experiments_v2.

Everything that must stay IDENTICAL to the v1 matched experiment is imported
from Matched_Experiments/matched_common.py, unchanged:
  * the 224x224 image cache (bit-identical to the original PIL pipeline)
  * the membership ground truth (train / val / non-member split)
  * the attacker's unlabelled pool and the 5k + 5k evaluation set
  * the victim architecture (DenseNet-121 + dropout head) and basic metrics

New in v2:
  * perturbation budgets are given in PIXEL units (ε/255) and converted per
    channel to the normalised input space; adversarial images are clamped to
    the valid image range (v1 used ε in normalised units and clamped to ±3)
  * FGSM, FGSM-RS (Wong et al., ICLR 2020), N-FGSM (de Jorge et al., NeurIPS
    2022) and PGD on that scale — device-agnostic and AMP-aware
  * the paper's shadow-model architectures (same heads as v1)
  * black-box victim queries on cached images, metric helpers, atomic I/O

Testing switches (never needed on Kaggle):
  V2_ALLOW_CPU=1      run on CPU when no GPU is present
  V2_NO_PRETRAINED=1  skip ImageNet weights (no download)
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

V2_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
MATCHED_DIR  = os.path.join(PROJECT_ROOT, "Matched_Experiments")
if MATCHED_DIR not in sys.path:
    sys.path.insert(0, MATCHED_DIR)
import matched_common as mc  # noqa: E402  (v1 shared module, imported unchanged)

NUM_CLASSES        = mc.NUM_CLASSES
DISEASE_CLASSES    = mc.DISEASE_CLASSES
DEFAULT_CACHE_DIR  = mc.DEFAULT_CACHE_DIR
PORTABLE_MANIFEST  = os.path.join(MATCHED_DIR, "kaggle", "manifest_portable.csv.gz")
BUILD_CACHE_SCRIPT = os.path.join(MATCHED_DIR, "build_image_cache.py")

NO_PRETRAINED = os.environ.get("V2_NO_PRETRAINED") == "1"
ALLOW_CPU     = os.environ.get("V2_ALLOW_CPU") == "1"
CHANNELS_LAST = torch.channels_last


# ─── Victim cells and run names ───────────────────────────────────────────────

# FGSM victim recipe (cell D of v2).  v1 used plain FGSM, ε = 0.02 in
# normalised units (≈ 1.15/255 in pixels), α = 0.5.
DEFAULT_METHOD = "fgsm_rs"
DEFAULT_EPS255 = 4.0
DEFAULT_ALPHA  = 0.5
METHODS        = ("fgsm_rs", "nfgsm", "fgsm", "pgd3")   # pgd3: 3-step PGD (≈1.6× the cost)

CELLS = {
    "std_noreg":  dict(training="standard", tag="C",
                       desc="Standard loss, no augmentation, no early stopping (35 ep, last)"),
    "fgsm_noreg": dict(training="fgsm", tag="D",
                       desc="FGSM adversarial training, no augmentation, no early stopping "
                            "(35 ep, last)"),
}


def victim_cell_id(cell, method=DEFAULT_METHOD, eps255=DEFAULT_EPS255, alpha=DEFAULT_ALPHA):
    """Folder-safe id of a victim recipe.  The default FGSM recipe is plain
    'fgsm_noreg'; any other FGSM recipe gets its settings in the name."""
    if cell == "std_noreg":
        return cell
    if (method, float(eps255), float(alpha)) == (DEFAULT_METHOD, DEFAULT_EPS255, DEFAULT_ALPHA):
        return "fgsm_noreg"
    return f"fgsm_noreg-{method}-eps{float(eps255):g}-a{float(alpha):g}"


def run_name(cell_id, seed):
    return f"{cell_id}_seed{seed}"


def parse_run(run):
    cell_id, seed = run.rsplit("_seed", 1)
    return cell_id, int(seed)


def parse_fgsm_variant(spec):
    """'fgsm_rs:4:0.5' → ('fgsm_rs', 4.0, 0.5)."""
    parts = spec.split(":")
    if len(parts) != 3 or parts[0] not in METHODS:
        raise ValueError(f"bad FGSM variant '{spec}' — use method:eps255:alpha, "
                         f"method in {METHODS}, e.g. fgsm_rs:4:0.5")
    return parts[0], float(parts[1]), float(parts[2])


# ─── Device / AMP ─────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if ALLOW_CPU:
        return torch.device("cpu")
    sys.exit("ERROR: CUDA is required (V2_ALLOW_CPU=1 is only for local pipeline tests).")


def amp_ctx(device, amp=True):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=bool(amp and device.type == "cuda"))


def make_generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def pin(device):
    return device.type == "cuda"


# ─── Perturbations in pixel units ─────────────────────────────────────────────

_MEAN = torch.tensor(mc.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_STD  = torch.tensor(mc.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
_LO   = (0.0 - _MEAN) / _STD            # normalised value of a black pixel, per channel
_HI   = (1.0 - _MEAN) / _STD            # normalised value of a white pixel, per channel
AMP_INPUT_GRAD_SCALE = 2.0 ** 14        # keeps fp16 input gradients out of underflow


def eps_t(eps255, ref):
    """ε/255 in pixel space → per-channel tensor (1,3,1,1) in normalised space."""
    return (float(eps255) / 255.0 / _STD).to(device=ref.device, dtype=ref.dtype)


def eps_normalised(eps255):
    return [float(eps255) / 255.0 / s for s in mc.IMAGENET_STD]


def clamp_valid(x):
    """Clamp a normalised image to the range of a valid [0, 1] image."""
    return torch.max(torch.min(x, _HI.to(x)), _LO.to(x))


def _project(delta, e):
    return torch.max(torch.min(delta, e), -e)


def input_grad(model, x, y, criterion, amp=True):
    """∇x criterion(model(x), y) — input gradient only (no parameter grads kept)."""
    x = x.detach().clone().requires_grad_(True)
    with amp_ctx(x.device, amp):
        logits = model(x)
    loss = criterion(logits.float(), y)
    if amp and x.device.type == "cuda":
        loss = loss * AMP_INPUT_GRAD_SCALE           # the sign is unaffected by the scale
    (g,) = torch.autograd.grad(loss, x)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def _uniform(x, scale, generator):
    u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (u * 2.0 - 1.0) * scale


def fgsm(model, x, y, eps255, criterion, amp=True):
    """x + ε·sign(∇x L), clamped to the valid image range."""
    g = input_grad(model, x, y, criterion, amp)
    return clamp_valid(x.detach() + eps_t(eps255, x) * g.sign()).detach()


def fgsm_rs(model, x, y, eps255, criterion, amp=True, step_mult=1.25, generator=None):
    """FGSM with random start (Wong et al. 2020): δ0 ~ U(-ε, ε), one step of
    1.25·ε, projected back to the ε-ball."""
    x = x.detach()
    e = eps_t(eps255, x)
    x0 = clamp_valid(x + _uniform(x, e, generator))
    g = input_grad(model, x0, y, criterion, amp)
    delta = _project(x0 - x + step_mult * e * g.sign(), e)
    return clamp_valid(x + delta).detach()


def nfgsm(model, x, y, eps255, criterion, amp=True, noise_mult=2.0, generator=None):
    """N-FGSM (de Jorge et al. 2022): δ0 ~ U(-2ε, 2ε), one step of ε, no projection."""
    x = x.detach()
    e = eps_t(eps255, x)
    x0 = clamp_valid(x + _uniform(x, noise_mult * e, generator))
    g = input_grad(model, x0, y, criterion, amp)
    return clamp_valid(x0 + e * g.sign()).detach()


def pgd(model, x, y, eps255, steps, criterion, amp=True, step255=None, random_start=True,
        generator=None):
    """L∞ PGD with step ε/4 by default (Madry et al. 2018)."""
    x = x.detach()
    e = eps_t(eps255, x)
    a = eps_t(step255 if step255 is not None else float(eps255) / 4.0, x)
    xa = clamp_valid(x + _uniform(x, e, generator)) if random_start else x.clone()
    for _ in range(int(steps)):
        g = input_grad(model, xa, y, criterion, amp)
        xa = clamp_valid(x + _project(xa - x + a * g.sign(), e)).detach()
    return xa


def make_train_attack(method):
    """Training-time attack: f(model, x, y, eps255, criterion, amp, generator)."""
    if method == "fgsm":
        return lambda m, x, y, e, c, amp, gen: fgsm(m, x, y, e, c, amp)
    if method == "fgsm_rs":
        return lambda m, x, y, e, c, amp, gen: fgsm_rs(m, x, y, e, c, amp, generator=gen)
    if method == "nfgsm":
        return lambda m, x, y, e, c, amp, gen: nfgsm(m, x, y, e, c, amp, generator=gen)
    if method == "pgd3":     # Madry-style PGD, 3 steps of 2.5·ε/3, random start
        return lambda m, x, y, e, c, amp, gen: pgd(m, x, y, e, 3, c, amp, step255=2.5 * e / 3,
                                                    generator=gen)
    raise ValueError(f"unknown method {method}")


def fgsm_v1(x, grad_sign, eps_norm=0.01):
    """The paper's AFMIA probe: ε in normalised units, clamp to ±3 (v1, unchanged)."""
    return torch.clamp(x + eps_norm * grad_sign, -3.0, 3.0)


# ─── Models ───────────────────────────────────────────────────────────────────

def build_victim(pretrained=None):
    """DenseNet-121 + Linear-ReLU-Dropout(0.3)-Linear head (identical to v1)."""
    return mc.build_victim_model(NUM_CLASSES, pretrained=(not NO_PRETRAINED)
                                 if pretrained is None else pretrained)


def load_victim(path, device):
    m = mc.build_victim_model(NUM_CLASSES, pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).to(memory_format=CHANNELS_LAST).eval()


SHADOW_ARCHS = ["resnet18", "mobilenet_v3_small", "efficientnet_b0", "densenet121",
                "shufflenet_v2_x1_0"]


def build_shadow(arch, seed, pretrained=None):
    """Same architectures and heads as Membership_Inference_Attack/shadow_models.py."""
    pretrained = (not NO_PRETRAINED) if pretrained is None else pretrained
    torch.manual_seed(int(seed))
    w = (lambda weights: weights if pretrained else None)
    if arch == "resnet18":
        m = models.resnet18(weights=w(models.ResNet18_Weights.IMAGENET1K_V1))
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=w(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1))
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w(models.EfficientNet_B0_Weights.IMAGENET1K_V1))
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif arch == "densenet121":
        m = models.densenet121(weights=w(models.DenseNet121_Weights.IMAGENET1K_V1))
        m.classifier = nn.Sequential(nn.Linear(m.classifier.in_features, 512), nn.ReLU(),
                                     nn.Linear(512, NUM_CLASSES))
    elif arch == "shufflenet_v2_x1_0":
        m = models.shufflenet_v2_x1_0(weights=w(models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1))
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"unknown shadow architecture {arch}")
    return m


def load_shadow(arch, path, device):
    m = build_shadow(arch, seed=0, pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).eval()


def pos_weight_from(labels):
    """clip(num_neg / num_pos, 0.1, 10) — the formula of every v1 training script."""
    labels = np.asarray(labels, dtype=np.float32)
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.1, 10.0), dtype=torch.float32)


# ─── Data ─────────────────────────────────────────────────────────────────────

def loader(cache, indices, labels=None, batch_size=128, shuffle=False, workers=4, device=None,
           generator=None):
    n_workers = mc.loader_workers(workers)
    return DataLoader(mc.CachedDataset(cache, indices, labels), batch_size=batch_size,
                      shuffle=shuffle, num_workers=n_workers,
                      pin_memory=bool(device is not None and pin(device)),
                      persistent_workers=False, generator=generator)


def labels_for_paths(paths):
    """True labels of attack-split images (train or non-member split)."""
    train_df, _, nonmember_df = mc.load_splits()
    table = {os.path.basename(p): y for df in (train_df, nonmember_df)
             for p, y in zip(df["path"], mc.parse_labels(df))}
    return np.stack([table[os.path.basename(p)] for p in paths]).astype(np.float32)


def attack_split(seed, smoke=False):
    """Attacker pool + 5k/5k evaluation set — exactly v1's (matched_common)."""
    return mc.attack_split_for_seed(seed, smoke=smoke)


@torch.no_grad()
def predict_probs(model, cache, indices, device, batch_size=128, amp=False, workers=4,
                  channels_last=False):
    """Black-box query: sigmoid probabilities (N, 15) for cached images."""
    model.eval()
    out = []
    for x in loader(cache, indices, batch_size=batch_size, workers=workers, device=device):
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=CHANNELS_LAST)
        with amp_ctx(device, amp):
            logits = model(x)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, NUM_CLASSES), np.float32)


# ─── Metrics ──────────────────────────────────────────────────────────────────

P_CLIP = 1e-7


def logit_np(p):
    p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1.0 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def bce_np(p, y):
    """Element-wise binary cross-entropy of probabilities p against labels y."""
    p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1.0 - P_CLIP)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def acc_at_median(y_true, scores):
    """Prior-matched accuracy: the top half of the scores is called 'member'.
    Legitimate on a balanced evaluation set, where the attacker knows that
    half of the candidates are members (no evaluation labels are used).
    Ties at the cut are broken at random — returned as the exact expectation,
    so a constant score gives 0.5 whatever the order of the images."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    k = len(y) // 2
    if k == 0:
        return float("nan")
    t = np.sort(s)[::-1][k - 1]                    # k-th largest score
    above, tied = s > t, s == t
    r = k - int(above.sum())                       # tied images still to call 'member'
    q = r / int(tied.sum())                        # probability a tied image is called 'member'
    correct = (y[above] == 1).sum() + (y[~above & ~tied] == 0).sum() \
        + q * (y[tied] == 1).sum() + (1 - q) * (y[tied] == 0).sum()
    return float(correct / len(y))


def score_metrics(y_true, scores):
    """Threshold-free metrics for a membership score (higher = member)."""
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    if not finite.all():
        s = np.where(finite, s, s[finite].min() if finite.any() else 0.0)
    m = mc.membership_scores_metrics(s[np.asarray(y_true) == 1], s[np.asarray(y_true) == 0])
    m["acc_median"] = acc_at_median(y_true, s)
    return m


# ─── I/O ──────────────────────────────────────────────────────────────────────

def write_json(path, obj):
    mc.write_json(path, obj)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save_npy(path, arr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def torch_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def victim_pool_probs(attack_dir, victim_dir, cache, pool_idx, device):
    """The attacker's black-box queries of the victim on the whole pool
    (pseudo-labels for the shadows).  Computed once, shared by both shadow sets."""
    path = os.path.join(attack_dir, "victim_pool_probs.npy")
    if os.path.exists(path):
        probs = np.load(path)
        if len(probs) == len(pool_idx):
            return probs
    victim = load_victim(os.path.join(victim_dir, "model.pth"), device)
    probs = predict_probs(victim, cache, pool_idx, device, batch_size=128, amp=False,
                          channels_last=True).astype(np.float32)
    save_npy(path, probs)
    del victim
    return probs
