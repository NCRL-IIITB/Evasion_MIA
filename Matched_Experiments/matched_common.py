"""
matched_common.py
=================
Shared building blocks for the matched-control experiments (reviewer concern:
"the baseline uses augmentation + early stopping, the adversarial model does
not — a consistent training setup is needed").

Everything that must be IDENTICAL across the four cells of the 2x2 design
lives here, so no cell can silently diverge:

  * the image source           — a 224x224 uint8 cache built once from the NIH
                                 zips (bit-identical to the original
                                 PIL Resize → ToTensor → Normalize pipeline,
                                 verified by build_image_cache.py)
  * the membership ground truth — train / val / non-member splits, replicated
                                 exactly from the original training scripts
  * the attacker's pool / eval  — replicated exactly from the original MIA
                                 runners, but members are drawn ONLY from the
                                 images the victim actually trained on
  * the victim architecture, FGSM, and the attack metrics

The original attack code (Membership_Inference_Attack/, Fluctuation_MIA/) is
imported unchanged.  Only its image-loading Dataset classes are redirected to
the cache (patch_original_attack_io), which changes I/O speed, not the inputs.
"""

import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score, roc_curve


# ─── Paths ────────────────────────────────────────────────────────────────────

MATCHED_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(MATCHED_DIR)
VICTIM_DIR    = os.path.join(PROJECT_ROOT, "Victim_Model")
MIA_DIR       = os.path.join(PROJECT_ROOT, "Membership_Inference_Attack")
FLUCT_DIR     = os.path.join(PROJECT_ROOT, "Fluctuation_MIA")
# MATCHED_MANIFEST overrides the manifest for pipeline tests only
MANIFEST_PATH = os.environ.get("MATCHED_MANIFEST", os.path.join(VICTIM_DIR, "manifest.csv"))
ZIP_DIR       = os.path.join(VICTIM_DIR, "data")
DEFAULT_CACHE_DIR = os.path.join(MATCHED_DIR, "cache")

# ─── Constants shared with the original code ──────────────────────────────────

DISEASE_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema",   "Fibrosis",       "Effusion",     "Pneumonia",    "Pleural_Thickening",
    "Cardiomegaly","Nodule",          "Mass",         "Hernia",       "No Finding",
]
NUM_CLASSES   = len(DISEASE_CLASSES)  # 15
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224

VAL_FRACTION  = 0.15   # same as train_baseline.py / train_adversarial.py
SPLIT_SEED    = 42     # fixed for every victim → identical membership ground truth

# Attacker pool / eval sizes — same as run_original_mia.py / run_fluctuation_attack.py
NUM_POOL_MEMBERS    = 20_000
NUM_POOL_NONMEMBERS = 20_000
NUM_EVAL_MEMBERS    = 5_000
NUM_EVAL_NONMEMBERS = 5_000

# The four cells of the 2x2 design (+ the robust-early-stopping variant of B)
CELLS = {
    "std_reg":           dict(training="standard", regime="reg",   tag="A",
                              desc="Standard loss, augmentation + early stopping (clean val AUC)"),
    "fgsm_reg":          dict(training="fgsm",     regime="reg",   tag="B",
                              desc="FGSM-AT, augmentation + early stopping (clean val AUC)"),
    "fgsm_reg_robustES": dict(training="fgsm",     regime="reg",   tag="B2",
                              desc="FGSM-AT, augmentation + early stopping (adversarial val AUC)"),
    "std_noreg":         dict(training="standard", regime="noreg", tag="C",
                              desc="Standard loss, no augmentation, no early stopping (35 ep)"),
    "fgsm_noreg":        dict(training="fgsm",     regime="noreg", tag="D",
                              desc="FGSM-AT, no augmentation, no early stopping (35 ep)"),
}


def run_name(cell: str, seed: int) -> str:
    return f"{cell}_seed{seed}"


def loader_workers(default: int) -> int:
    """DataLoader worker count; MATCHED_NUM_WORKERS caps it (e.g. 4 vCPUs on Kaggle)."""
    cap = os.environ.get("MATCHED_NUM_WORKERS")
    return max(0, min(default, int(cap))) if cap else default


# ─── Image cache ──────────────────────────────────────────────────────────────

class ImageCache:
    """Read-only view of the 224x224 image cache built by build_image_cache.py.

    Images are stored as one grayscale uint8 plane (N, 224, 224) because NIH
    X-rays are grayscale; the rare image whose RGB channels differ after
    convert("RGB") + Resize is stored separately in full colour, so the cache
    reproduces the original pipeline exactly for every image.
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        info_path = os.path.join(cache_dir, "cache_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"{info_path} not found — run build_image_cache.py first."
            )
        with open(info_path) as f:
            self.info = json.load(f)
        if not self.info.get("complete", False):
            raise RuntimeError(f"Image cache in {cache_dir} is incomplete — rebuild it.")
        with open(os.path.join(cache_dir, "keys.json")) as f:
            self.keys = json.load(f)
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self._open()

    def _open(self):
        self.gray = np.load(os.path.join(self.cache_dir, "gray_224.npy"), mmap_mode="r")
        color_idx = self.info.get("color_indices", [])
        self.color_rows = {int(i): r for r, i in enumerate(color_idx)}
        self.color = (
            np.load(os.path.join(self.cache_dir, "color_224.npy"))
            if color_idx else None
        )

    # Pickle only the location (a memmap would otherwise be copied in full)
    def __getstate__(self):
        return {"cache_dir": self.cache_dir, "info": self.info, "keys": self.keys}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self._open()

    def __len__(self):
        return len(self.keys)

    def index_of(self, paths) -> np.ndarray:
        """Map image paths (any directory) → cache indices, keyed by file name."""
        return np.fromiter(
            (self.key_to_idx[os.path.basename(str(p))] for p in paths),
            dtype=np.int64, count=len(paths),
        )

    def uint8_chw(self, i: int) -> torch.Tensor:
        """(3, 224, 224) uint8 tensor — what PIL convert('RGB') + Resize gives."""
        i = int(i)
        if i in self.color_rows:
            arr = np.array(self.color[self.color_rows[i]])              # (H, W, 3)
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        g = torch.from_numpy(np.array(self.gray[i]))                    # (H, W)
        return g.unsqueeze(0).expand(3, -1, -1).contiguous()

    def normalized(self, i: int) -> torch.Tensor:
        """Same values as transforms.ToTensor() + Normalize(ImageNet)."""
        return normalize_uint8(self.uint8_chw(i))


def normalize_uint8(x: torch.Tensor) -> torch.Tensor:
    """uint8 (3,H,W) → float, exactly as ToTensor().div(255) + Normalize."""
    return TF.normalize(x.float().div(255), IMAGENET_MEAN, IMAGENET_STD)


_STANDARD_TRANSFORM_REPR = repr(transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
]))


def _check_standard_transform(transform):
    """The cache replaces only the plain inference transform; refuse anything else."""
    if transform is not None and repr(transform) != _STANDARD_TRANSFORM_REPR:
        raise ValueError(
            "Cached datasets only replace the plain Resize+ToTensor+Normalize "
            f"transform, got:\n{transform!r}"
        )


class CachedDataset(Dataset):
    """Cache-backed dataset: returns (image, label) or just image.

    ``uint8_transform`` (optional) runs on the uint8 (3,224,224) tensor before
    normalisation — used for training-time augmentation.
    """

    def __init__(self, cache: ImageCache, indices, labels=None, uint8_transform=None):
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.uint8_transform = uint8_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.cache.uint8_chw(self.indices[i])
        if self.uint8_transform is not None:
            x = self.uint8_transform(x)
        x = normalize_uint8(x)
        if self.labels is None:
            return x
        return x, torch.from_numpy(self.labels[i])


# ─── Manifest, membership splits, attacker pool / eval ───────────────────────

def parse_labels(df: pd.DataFrame) -> np.ndarray:
    return np.array([ast.literal_eval(s) for s in df["label_idx"]], dtype=np.float32)


def load_splits(manifest_path: str = MANIFEST_PATH):
    """Return (train_df, val_df, nonmember_df).

    Replicates train_baseline.py / train_adversarial.py exactly: members are
    shuffled with random_state=42, the first 15 % become the validation set
    (used only for early stopping / monitoring), the rest is the training set.

    IMPORTANT: only train_df images are members of the victim's training set.
    The original MIA runners drew "members" from the whole 78,708-image member
    split, including the 11,806 validation images that were never trained on.
    With seed 42 this was harmless: pandas' shuffle and the runner's
    permutation consume the same RandomState(42) sequence, so every
    validation image landed in the attacker's unlabelled pool and none in the
    5k evaluation members.  With any other seed (43, 44, …) validation images
    would be scored as members, so members are now drawn from train_df only.
    """
    manifest  = pd.read_csv(manifest_path)
    member_df = manifest[manifest["split"] == "member"].reset_index(drop=True)
    n_val     = int(len(member_df) * VAL_FRACTION)
    shuffled  = member_df.sample(frac=1.0, random_state=SPLIT_SEED).reset_index(drop=True)
    val_df    = shuffled.iloc[:n_val].reset_index(drop=True)
    train_df  = shuffled.iloc[n_val:].reset_index(drop=True)
    nonmember_df = manifest[manifest["split"] == "nonmember"].reset_index(drop=True)
    return train_df, val_df, nonmember_df


def build_attack_split(member_paths_all, nonmember_paths_all, seed: int,
                       n_pool_m=NUM_POOL_MEMBERS, n_pool_nm=NUM_POOL_NONMEMBERS,
                       n_eval_m=NUM_EVAL_MEMBERS, n_eval_nm=NUM_EVAL_NONMEMBERS):
    """Attacker's shuffled unlabelled pool + labelled eval sets.

    Same procedure as build_pool_and_eval() in run_original_mia.py and
    run_fluctuation_attack.py (seed=42 reproduces their sampling order).
    """
    member_paths_all    = np.asarray(member_paths_all, dtype=object)
    nonmember_paths_all = np.asarray(nonmember_paths_all, dtype=object)
    rng = np.random.RandomState(seed)

    max_pool_m  = max(0, len(member_paths_all)    - n_eval_m)
    max_pool_nm = max(0, len(nonmember_paths_all) - n_eval_nm)
    n_pool_m    = min(n_pool_m,  max_pool_m)
    n_pool_nm   = min(n_pool_nm, max_pool_nm)

    member_idx    = rng.permutation(len(member_paths_all))
    nonmember_idx = rng.permutation(len(nonmember_paths_all))

    n_eval_m  = min(n_eval_m,  len(member_paths_all)    - n_pool_m)
    n_eval_nm = min(n_eval_nm, len(nonmember_paths_all) - n_pool_nm)

    pool_member_paths    = member_paths_all[member_idx[:n_pool_m]]
    pool_nonmember_paths = nonmember_paths_all[nonmember_idx[:n_pool_nm]]
    eval_member_paths    = member_paths_all[member_idx[n_pool_m: n_pool_m + n_eval_m]]
    eval_nonmember_paths = nonmember_paths_all[nonmember_idx[n_pool_nm: n_pool_nm + n_eval_nm]]

    pool_all = np.concatenate([pool_member_paths, pool_nonmember_paths])
    rng.shuffle(pool_all)
    return pool_all, eval_member_paths, eval_nonmember_paths


def attack_split_for_seed(seed: int, smoke: bool = False):
    """Attack split used by EVERY attack on every victim trained with ``seed``."""
    train_df, _, nonmember_df = load_splits()
    sizes = {}
    if smoke:
        sizes = dict(n_pool_m=400, n_pool_nm=400, n_eval_m=100, n_eval_nm=100)
    return build_attack_split(
        train_df["path"].values, nonmember_df["path"].values, seed, **sizes
    )


def save_attack_split(path: str, seed: int, pool, eval_m, eval_nm):
    """Write the attack split once per seed (audit trail; never overwritten)."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"   # several attack jobs may race here
    with open(tmp, "w") as f:
        json.dump({
            "seed": seed,
            "member_definition": "images in the victim's training set only "
                                 "(validation images excluded)",
            "nonmember_definition": "manifest 'nonmember' split (patient-disjoint)",
            "pool": [os.path.basename(p) for p in pool],
            "eval_members": [os.path.basename(p) for p in eval_m],
            "eval_nonmembers": [os.path.basename(p) for p in eval_nm],
        }, f)
    os.replace(tmp, path)


# ─── Victim model ─────────────────────────────────────────────────────────────

def build_victim_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """DenseNet-121 + Linear(1024,512)-ReLU-Dropout(0.3)-Linear(512,15).

    Same architecture as train_baseline.py / train_adversarial.py and the
    head that Victim_Model/api.py rebuilds when meta['dropout'] is True.
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(512, num_classes)
    )
    for p in model.parameters():
        p.requires_grad = True
    return model


def load_victim_weights(path: str, device, num_classes: int = NUM_CLASSES) -> nn.Module:
    model = build_victim_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


# ─── FGSM ─────────────────────────────────────────────────────────────────────

AMP_INPUT_GRAD_SCALE = 2.0 ** 14   # keeps fp16 input gradients out of underflow


def fgsm(model, x, y, epsilon, criterion, amp=False):
    """x_adv = clamp(x + ε·sign(∇x L(f(x), y)), -3, 3) — the paper's Eq. (2).

    Only the input gradient is computed (parameter gradients are not needed and
    are not accumulated).  With amp=True the loss is scaled before backward so
    fp16 gradients do not underflow; the sign is unaffected by the scale.
    """
    x = x.detach().clone().requires_grad_(True)
    if amp:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss = criterion(logits.float(), y) * AMP_INPUT_GRAD_SCALE
    else:
        loss = criterion(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    grad_sign = torch.nan_to_num(grad.sign(), nan=0.0)
    return torch.clamp(x.detach() + epsilon * grad_sign, -3.0, 3.0)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def macro_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Macro AUC over classes that have both labels present.

    Identical to roc_auc_score(average="macro") whenever every class occurs
    (always true for the full 11,806-image validation set); avoids NaN on
    small subsets where a rare class (e.g. Hernia) may be absent.
    """
    present = (y_true.min(axis=0) == 0) & (y_true.max(axis=0) == 1)
    if not present.any():
        return float("nan")
    return float(roc_auc_score(y_true[:, present], probs[:, present], average="macro"))


def tpr_at_fpr(y_true, scores, target_fpr: float) -> float:
    """Highest TPR achievable at FPR ≤ target (Carlini et al., S&P 2022)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def membership_scores_metrics(scores_members, scores_nonmembers) -> dict:
    """Threshold-free metrics for a score where HIGHER = more likely member."""
    y = np.concatenate([np.ones(len(scores_members)), np.zeros(len(scores_nonmembers))])
    s = np.concatenate([scores_members, scores_nonmembers]).astype(np.float64)
    finite = np.isfinite(s)
    if not finite.all():   # non-finite score → least member-like
        s = np.where(finite, s, s[finite].min() if finite.any() else 0.0)
    fpr, tpr, _ = roc_curve(y, s)
    # Best balanced accuracy over thresholds (optimistic, oracle threshold)
    best_acc = float(np.max(0.5 * (tpr + (1.0 - fpr))))
    return {
        "auc":            float(roc_auc_score(y, s)),
        "tpr_at_1pct_fpr":   tpr_at_fpr(y, s, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
        "best_threshold_acc": best_acc,
    }


# ─── Redirect the ORIGINAL attack code's image loading to the cache ───────────

def import_original_attack_modules(afmia: bool):
    """Put the original attack directories on sys.path and import them.

    Membership_Inference_Attack/ provides mia.py, mia_variance.py,
    shadow_models.py; Fluctuation_MIA/ provides the AFMIA modules;
    Victim_Model/ provides api.py.  None of these files is modified.
    """
    for d in (VICTIM_DIR, MIA_DIR) + ((FLUCT_DIR,) if afmia else ()):
        if d not in sys.path:
            sys.path.insert(0, d)
    import mia, mia_variance, shadow_models, api  # noqa: F401
    if afmia:
        import shadow_models_fluctuation, mia_fluctuation, mia_fluctuation_variance  # noqa: F401


def patch_original_attack_io(cache: ImageCache):
    """Swap the Dataset classes the original attack code uses for cache-backed
    ones with the same constructor signatures and the same returned tensors.

    Patched names (looked up at call time inside the original modules):
      shadow_models.ShadowDataset / InferenceDataset       (shadow fit / predict)
      shadow_models_fluctuation.ShadowDataset / ...        (AFMIA shadow fit / FGSM)
      mia_fluctuation._PathImageDataset                    (AFMIA victim probing)
    """

    class CachedShadowDataset(Dataset):
        def __init__(self, paths, labels, transform):
            _check_standard_transform(transform)
            self.idx = cache.index_of(paths)
            self.labels = labels

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            return (cache.normalized(self.idx[i]),
                    torch.tensor(self.labels[i], dtype=torch.float32))

    class CachedInferenceDataset(Dataset):
        def __init__(self, paths, transform):
            _check_standard_transform(transform)
            self.idx = cache.index_of(paths)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            return cache.normalized(self.idx[i])

    import shadow_models
    shadow_models.ShadowDataset = CachedShadowDataset
    shadow_models.InferenceDataset = CachedInferenceDataset
    if "shadow_models_fluctuation" in sys.modules:
        smf = sys.modules["shadow_models_fluctuation"]
        smf.ShadowDataset = CachedShadowDataset
        smf.InferenceDataset = CachedInferenceDataset
    if "mia_fluctuation" in sys.modules:
        sys.modules["mia_fluctuation"]._PathImageDataset = CachedInferenceDataset
    return CachedInferenceDataset


def make_cached_victim_api(cache: ImageCache, inference_dataset_cls):
    """Subclass of the original VictimAPI whose predict() reads from the cache.

    The model, metadata handling and transform come from Victim_Model/api.py
    unchanged; only the inline PIL dataset inside predict() is replaced.
    """
    from api import VictimAPI

    class CachedVictimAPI(VictimAPI):
        def predict(self, X: np.ndarray) -> np.ndarray:
            loader = DataLoader(
                inference_dataset_cls(X, self.transform),
                batch_size=self.batch_size, shuffle=False,
                num_workers=loader_workers(4) if self.device.type == "cuda" else 0,
                pin_memory=(self.device.type == "cuda"),
            )
            out = []
            with torch.no_grad():
                for batch in loader:
                    logits = self.model(batch.to(self.device, non_blocking=True))
                    out.append(torch.sigmoid(logits).cpu().numpy())
            return np.vstack(out)

    return CachedVictimAPI


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)
