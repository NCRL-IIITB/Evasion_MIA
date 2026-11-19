"""E3 — Medical imaging datasets (MedMNIST v3 at 64x64).

Question: does CF-MIA beat the existing attacks on medical images — the paper's
domain — including ChestMNIST, which is the NIH ChestX-ray14 dataset of the paper
(112,120 frontal chest X-rays, 14 labels, multi-label)?

  chestmnist@64   NIH ChestX-ray14, multi-label (14), population 30k of 100k
  dermamnist@64   HAM10000 dermatoscopy, 7 classes, ~8k images (small: strong memorisation)
  pathmnist@64    NCT-CRC-HE colon histology, 9 classes, population 30k of 100k
  bloodmnist@64   blood-cell microscopy, 8 classes, ~13.6k images

Needs `pip install medmnist` (the notebook does it) or the .npz files attached as
a Kaggle input (named e.g. chestmnist_64.npz).
"""
EXPERIMENT = "E3_medical_datasets"
DESCRIPTION = """
**E3 — medical datasets.** Four MedMNIST v3 datasets at 64×64: ChestMNIST (= NIH ChestX-ray14,
the paper's dataset; multi-label), DermaMNIST, PathMNIST, BloodMNIST.  Victims: ResNet-18 with
standard training (`*_r18_std`) or N-FGSM adversarial training at ε = 4/255 (`*_r18_at`);
VGG-11 families serve as black-box references.  Population = official train + val (capped at
30k), each model trains on a random half; the official test split measures accuracy and evasion.
"""

DATASETS = {
    "chestmnist@64": dict(n_pop=30000, n_test=10000, extract_bs=400, evasion_eps=[1, 4], robust_eps=4),
    "dermamnist@64": dict(n_pop=None, n_test=None, extract_bs=400, evasion_eps=[1, 4], robust_eps=4),
    "pathmnist@64": dict(n_pop=30000, n_test=None, extract_bs=400, evasion_eps=[1, 4], robust_eps=4),
    "bloodmnist@64": dict(n_pop=None, n_test=None, extract_bs=400, evasion_eps=[1, 4], robust_eps=4),
}
SHORT = {"chestmnist@64": "chest", "dermamnist@64": "derma", "pathmnist@64": "path", "bloodmnist@64": "blood"}

TRAIN = dict(epochs=40, batch=128, lr=0.05, wd=5e-4, aug=True)
AT = dict(method="nfgsm", eps=4)

FAMILIES = []
for _ds, _s in SHORT.items():
    for _t, _tr in (("std", {}), ("at", AT)):
        FAMILIES.append(dict(name=f"{_s}_r18_{_t}", dataset=_ds, arch="resnet18", train=dict(_tr),
                             n_models=10, n_sources=2))
        FAMILIES.append(dict(name=f"{_s}_vgg_{_t}", dataset=_ds, arch="vgg11", train=dict(_tr),
                             n_models=6, n_sources=2))
# Optional, closest to the paper (DenseNet-121, ImageNet-pretrained) — about 4x the cost of
# ResNet-18 at 64 px.  Uncomment to add it for ChestMNIST:
# for _t, _tr in (("std", {}), ("at", AT)):
#     FAMILIES.append(dict(name=f"chest_dn121_{_t}", dataset="chestmnist@64", arch="densenet121_pt",
#                          train=dict(_tr, opt="adamw", lr=1e-3, wd=1e-4, epochs=20), n_models=8, n_sources=0))


def _probes(s):
    return [{"kind": "hflip"}, {"kind": "augset"},
            {"kind": "gauss", "sigma": [0.01, 0.03], "k": 8},
            {"kind": "self_fgsm", "eps": [1, 2, 4], "vec": [1]},
            {"kind": "self_pgd", "eps": [4], "steps": 10},
            {"kind": "tr_fgsm", "src": f"{s}_r18_std", "eps": [1, 2, 4], "vec": [1]},
            {"kind": "tr_fgsm", "src": f"{s}_r18_at", "eps": [1, 2, 4], "vec": [1]},
            {"kind": "tr_fgsm", "src": f"{s}_vgg_std", "eps": [1, 4]},
            {"kind": "tr_fgsm", "src": f"{s}_vgg_at", "eps": [1, 4]}]


PROBES = {ds: _probes(s) for ds, s in SHORT.items()}


def _settings():
    out = []
    for ds, s in SHORT.items():
        for t, o in (("std", "at"), ("at", "std")):
            F = f"{s}_r18_{t}"
            out += [
                dict(name=f"GB_{F}", access="grey", target=F, refs=[F], sources=[F]),
                dict(name=f"WB_{F}", access="white", target=F, refs=[F], sources=[F], white_box=True),
                dict(name=f"GB_{F}_recipeX", access="grey", target=F, refs=[f"{s}_r18_{o}"],
                     sources=[f"{s}_r18_{o}"], note="Architecture known, training recipe wrong."),
                dict(name=f"BB_{F}", access="black", target=F, refs=[f"{s}_vgg_std", f"{s}_vgg_at"],
                     sources=[f"{s}_vgg_std", f"{s}_vgg_at"],
                     note="Unknown architecture and recipe: VGG-11 references of both recipes."),
            ]
    for x in out:
        x["n_targets"] = 3
    return out


SETTINGS = _settings()
SMOKE = dict(n_pop=1500, n_test=500, epochs=2, max_models=5)
REPORT = {}
