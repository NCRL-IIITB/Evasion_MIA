"""E2 — Generalisation across datasets and victim architectures (CIFAR-10, CIFAR-100).

Question: does CF-MIA beat the existing attacks for every victim architecture and
for a harder dataset (CIFAR-100, 100 classes, stronger memorisation), both for
standard and adversarially trained victims?
"""
EXPERIMENT = "E2_cifar_models"
DESCRIPTION = """
**E2 — datasets × architectures.** CIFAR-10 with ResNet-18, WRN-16-4 and VGG-11; CIFAR-100 with
ResNet-18 and WRN-16-4; each with standard training (`*_std`) and N-FGSM adversarial training
at ε = 8/255 (`*_at`).  Ten models per family (two are surrogate "sources"), each trained on a
random half of the 50k training images.  Per family: grey-box (same architecture and recipe),
white-box (victim gradients) and black-box (references of the other architectures).
"""

DATASETS = {
    "cifar10": dict(n_pop=None, n_test=None, extract_bs=500, evasion_eps=[2, 8], robust_eps=8),
    "cifar100": dict(n_pop=None, n_test=None, extract_bs=500, evasion_eps=[2, 8], robust_eps=8),
}

TRAIN = dict(epochs=50, batch=256, lr=0.1, wd=5e-4, aug=True)
AT = dict(method="nfgsm", eps=8)
ARCHS = {"c10": [("r18", "resnet18"), ("wrn", "wrn16_4"), ("vgg", "vgg11")],
         "c100": [("r18", "resnet18"), ("wrn", "wrn16_4")]}
DS_OF = {"c10": "cifar10", "c100": "cifar100"}

FAMILIES = []
for _d, _archs in ARCHS.items():
    for _short, _arch in _archs:
        for _t, _tr in (("std", {}), ("at", AT)):
            FAMILIES.append(dict(name=f"{_d}_{_short}_{_t}", dataset=DS_OF[_d], arch=_arch, train=dict(_tr),
                                 n_models=10, n_sources=2))


def _probes(d):
    fams = [f["name"] for f in FAMILIES if f["name"].startswith(d + "_")]
    return ([{"kind": "hflip"}, {"kind": "augset"},
             {"kind": "gauss", "sigma": [0.02], "k": 8},
             {"kind": "self_fgsm", "eps": [2, 8], "vec": [2]},
             {"kind": "self_pgd", "eps": [8], "steps": 10}] +
            [{"kind": "tr_fgsm", "src": f, "eps": [2, 8], "vec": [2]} for f in fams])


PROBES = {DS_OF[d]: _probes(d) for d in ARCHS}


def _settings():
    out = []
    for d, archs in ARCHS.items():
        for short, _ in archs:
            for t in ("std", "at"):
                F = f"{d}_{short}_{t}"
                others = [f"{d}_{o}_{t}" for o, _ in archs if o != short]
                out += [
                    dict(name=f"GB_{F}", access="grey", target=F, refs=[F], sources=[F], detail=False),
                    dict(name=f"WB_{F}", access="white", target=F, refs=[F], sources=[F], white_box=True,
                         detail=False),
                    dict(name=f"BB_{F}", access="black", target=F, refs=others, sources=others[:1], detail=False,
                         note=f"References: {', '.join(others)}; transfer surrogates: {others[0]}."),
                ]
    for s in out:
        s["n_targets"] = 3
    return out


SETTINGS = _settings()
SMOKE = dict(n_pop=1500, n_test=500, epochs=2, max_models=5)
REPORT = {}
