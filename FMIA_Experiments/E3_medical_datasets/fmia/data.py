"""Datasets.  Every dataset is returned as uint8 image tensors in (N, C, H, W) and
labels, split into
  * the POPULATION — the candidate records of the membership game.  Every zoo
    model trains on a random half of it (its `keep` mask), so each record is a
    member of about half of the models (LiRA / RMIA protocol);
  * a disjoint TEST split — only for clean accuracy and evasion.

Supported names
  cifar10, cifar100                       torchvision (downloads, ~170 MB)
  <flag>[@size] for MedMNIST v3           e.g. chestmnist@64, dermamnist@64, pathmnist@64,
                                          bloodmnist@64, organamnist@64 (pip install medmnist)
  synthetic, synthml                      tiny learnable datasets (multi-class / multi-label,
                                          1-channel) for CPU smoke tests
"""
import glob
import os

import numpy as np
import torch

from .utils import log, stable_seed

DATA_ROOT = os.environ.get("FMIA_DATA", "/tmp/fmia_data")
MULTILABEL_FLAGS = {"chestmnist"}


class Dataset:
    def __init__(self, key, x_pop, y_pop, x_test, y_test, num_classes, multilabel):
        self.key = key
        self.x_pop, self.y_pop = x_pop, y_pop
        self.x_test, self.y_test = x_test, y_test
        self.num_classes = int(num_classes)
        self.multilabel = bool(multilabel)
        self.in_ch = int(x_pop.shape[1])
        self.img_size = int(x_pop.shape[2])
        xf = x_pop[: min(len(x_pop), 5000)].float() / 255.0
        self.mean = xf.mean(dim=(0, 2, 3)).tolist()
        self.std = xf.std(dim=(0, 2, 3)).clamp_min(1e-3).tolist()

    @property
    def n_pop(self):
        return len(self.x_pop)

    def describe(self):
        return (f"{self.key}: population {self.n_pop}, test {len(self.x_test)}, "
                f"{self.in_ch}x{self.img_size}x{self.img_size}, {self.num_classes} "
                f"{'labels (multi-label)' if self.multilabel else 'classes'}")


def parse_key(key):
    name, _, size = key.partition("@")
    return name.lower(), (int(size) if size else None)


# ─── loaders ──────────────────────────────────────────────────────────────────

def _find_input(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def _cifar(name):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    folder = "cifar-10-batches-py" if name == "cifar10" else "cifar-100-python"
    attached = _find_input([f"/kaggle/input/**/{folder}"])
    root = os.path.dirname(attached) if attached else DATA_ROOT
    tr = cls(root=root, train=True, download=attached is None)
    te = cls(root=root, train=False, download=attached is None)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))
    return (to_t(tr.data), torch.tensor(tr.targets, dtype=torch.long),
            to_t(te.data), torch.tensor(te.targets, dtype=torch.long),
            10 if name == "cifar10" else 100, False)


def _medmnist(flag, size):
    size = size or 28
    fname = f"{flag}.npz" if size == 28 else f"{flag}_{size}.npz"
    path = _find_input([f"/kaggle/input/**/{fname}", os.path.join(DATA_ROOT, fname)])
    if path is None:
        try:
            import medmnist
            from medmnist import INFO
        except ImportError as e:
            raise RuntimeError("MedMNIST needs `pip install medmnist` (or attach the .npz "
                               "file as a Kaggle input)") from e
        os.makedirs(DATA_ROOT, exist_ok=True)
        cls = getattr(medmnist, INFO[flag]["python_class"])
        kw = {} if size == 28 else {"size": size}
        cls(split="test", download=True, root=DATA_ROOT, **kw)
        path = _find_input([os.path.join(DATA_ROOT, fname), os.path.join(DATA_ROOT, f"{flag}*{size}*.npz")])
        if path is None:
            raise RuntimeError(f"MedMNIST download finished but {fname} is not in {DATA_ROOT}")
    log(f"  MedMNIST file: {path}")
    z = np.load(path)

    def imgs(a):
        a = a if a.ndim == 4 else a[..., None]
        return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))

    xs = [imgs(z[f"{s}_images"]) for s in ("train", "val", "test")]
    ys = [z[f"{s}_labels"] for s in ("train", "val", "test")]
    multilabel = flag in MULTILABEL_FLAGS
    if multilabel:
        ys = [torch.from_numpy(y.astype(np.float32)) for y in ys]
        num_classes = ys[0].shape[1]
    else:
        ys = [torch.from_numpy(y.reshape(-1).astype(np.int64)) for y in ys]
        num_classes = int(max(int(y.max()) for y in ys)) + 1
    x_pop, y_pop = torch.cat(xs[:2]), torch.cat(ys[:2])
    return x_pop, y_pop, xs[2], ys[2], num_classes, multilabel


def _synthetic_ml(n_total=4000, num_labels=6, size=32, seed=0):
    """Multi-label variant: each label adds its own template; 10 % of labels flipped."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (num_labels, 1, 8, 8))
    tmpl = np.kron(base, np.ones((1, 1, size // 8, size // 8)))
    y = (rng.random((n_total, num_labels)) < 0.3).astype(np.float32)
    x = 0.8 * np.einsum("nl,lchw->nchw", y, tmpl) + rng.normal(0, 1.0, (n_total, 1, size, size))
    x = np.clip((x - x.min()) / (x.max() - x.min()) * 255, 0, 255).astype(np.uint8)
    flip = rng.random(y.shape) < 0.10
    y[flip] = 1 - y[flip]
    n_test = n_total // 4
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    return xt[n_test:], yt[n_test:], xt[:n_test], yt[:n_test], num_labels, True


def _synthetic(n_total=4000, num_classes=10, size=32, seed=0):
    """Class templates + noise + 15 % label noise (so that memorisation exists)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (num_classes, 3, 8, 8))
    tmpl = np.kron(base, np.ones((1, 1, size // 8, size // 8)))
    y = rng.integers(0, num_classes, n_total)
    x = 0.6 * tmpl[y] + rng.normal(0, 1.0, (n_total, 3, size, size))
    x = np.clip((x - x.min()) / (x.max() - x.min()) * 255, 0, 255).astype(np.uint8)
    noisy = rng.random(n_total) < 0.15
    y[noisy] = rng.integers(0, num_classes, noisy.sum())
    n_test = n_total // 4
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.int64))
    return xt[n_test:], yt[n_test:], xt[:n_test], yt[:n_test], num_classes, False


def load_dataset(key, n_pop=None, n_test=None, seed=0):
    """Load (downloading if needed) and subsample deterministically."""
    name, size = parse_key(key)
    if name in ("cifar10", "cifar100"):
        x_pop, y_pop, x_te, y_te, nc, ml = _cifar(name)
    elif name == "synthetic":
        x_pop, y_pop, x_te, y_te, nc, ml = _synthetic(size=size or 32)
    elif name == "synthml":
        x_pop, y_pop, x_te, y_te, nc, ml = _synthetic_ml(size=size or 32)
    else:
        x_pop, y_pop, x_te, y_te, nc, ml = _medmnist(name, size)
    rng = np.random.default_rng(stable_seed("population", key, seed))
    if n_pop and n_pop < len(x_pop):
        idx = np.sort(rng.choice(len(x_pop), n_pop, replace=False))
        x_pop, y_pop = x_pop[idx], y_pop[idx]
    if n_test and n_test < len(x_te):
        idx = np.sort(rng.choice(len(x_te), n_test, replace=False))
        x_te, y_te = x_te[idx], y_te[idx]
    return Dataset(key, x_pop.contiguous(), y_pop.contiguous(), x_te.contiguous(),
                   y_te.contiguous(), nc, ml)


def prefetch(key):
    """Download a dataset once (in the driver) so that two workers never race."""
    name, size = parse_key(key)
    if name in ("cifar10", "cifar100"):
        _cifar(name)
    elif name not in ("synthetic", "synthml"):
        _medmnist(name, size)


# ─── GPU-side batching and augmentation ───────────────────────────────────────

def to_float(xb):
    return xb.float().div_(255.0)


def random_crop_flip(x, pad, flip=True):
    """Random translation by up to `pad` px (reflect padding) + horizontal flip, batched on GPU."""
    B, C, H, W = x.shape
    if pad > 0:
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
        i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        x = _crop(xp, i, j, H, W)
    if flip:
        m = torch.rand(B, device=x.device) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(3), x)
    return x


def _crop(xp, i, j, H, W):
    B, C = xp.shape[:2]
    ar_h = torch.arange(H, device=xp.device)
    ar_w = torch.arange(W, device=xp.device)
    rows = (i[:, None] + ar_h[None, :])                       # B, H
    cols = (j[:, None] + ar_w[None, :])                       # B, W
    out = xp.gather(2, rows[:, None, :, None].expand(B, C, H, xp.shape[3]))
    return out.gather(3, cols[:, None, None, :].expand(B, C, H, W))


def shift(x, dx, dy):
    """Deterministic translation by (dx, dy) px with reflect padding (LiRA-style query)."""
    if dx == 0 and dy == 0:
        return x
    s = max(abs(dx), abs(dy))
    H, W = x.shape[2:]
    xp = torch.nn.functional.pad(x, (s, s, s, s), mode="reflect")
    return xp[:, :, s + dy:s + dy + H, s + dx:s + dx + W]
