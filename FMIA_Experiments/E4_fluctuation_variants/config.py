"""E4 — Which implementation of fluctuation-based MIA works, and why (CIFAR-10, ResNet-18).

Ablations, all on the same models and the same cached signals:
  * probe type        transfer FGSM vs transfer PGD vs own-label (paper) vs Gaussian /
                      uniform noise vs white-box FGSM / PGD; ε from 0.5/255 to 16/255
  * statistic         logit drop φ_clean − φ_probe vs Σ|Δp| (paper) vs KL vs flips
                      (per-statistic rows in each setting's "single statistics" table)
  * scoring           raw threshold → per-example calibration (offline / online) →
                      multi-probe stacking (CF-MIA) → + confidence
  * classifier        the paper's GB on |Δp| vs GB on all fluctuation features vs CF-MIA
  * #references       2, 4, 8, 17 reference models (cost of the attack)
"""
EXPERIMENT = "E4_fluctuation_variants"
DESCRIPTION = """
**E4 — fluctuation variants.** CIFAR-10, ResNet-18, standard (`r18_std`) and N-FGSM ε = 8/255
(`r18_at`) victims, 20 models per family (2 surrogate sources + 18 targets/references).  Every
model is probed with a large probe bank; the settings below restrict which probes an attack may
use, so each row isolates one design choice of the fluctuation attack.
"""

DATASETS = {
    "cifar10": dict(n_pop=None, n_test=None, extract_bs=500, evasion_eps=[2, 8], robust_eps=8),
}

TRAIN = dict(epochs=50, batch=256, lr=0.1, wd=5e-4, aug=True)
AT = dict(method="nfgsm", eps=8)

FAMILIES = [
    dict(name="r18_std", dataset="cifar10", arch="resnet18", train={}, n_models=20, n_sources=2),
    dict(name="r18_at", dataset="cifar10", arch="resnet18", train=AT, n_models=20, n_sources=2),
]

PROBES = {
    "cifar10": [
        {"kind": "hflip"},
        {"kind": "augset"},
        {"kind": "gauss", "sigma": [0.005, 0.01, 0.02, 0.04], "k": 8},
        {"kind": "uniform", "eps": [4, 8], "k": 8},
        {"kind": "self_fgsm", "eps": [0.5, 1, 2, 4, 8, 16], "vec": [2]},
        {"kind": "self_fgsm_own", "eps": [2, 8]},
        {"kind": "self_pgd", "eps": [2, 8], "steps": 10},
        {"kind": "tr_fgsm", "src": "r18_std", "eps": [0.5, 1, 2, 4, 8, 16], "vec": [2]},
        {"kind": "tr_fgsm_own", "src": "r18_std", "eps": [2, 8], "vec": [2]},
        {"kind": "tr_pgd", "src": "r18_std", "eps": [4, 8], "steps": 5},
        {"kind": "tr_fgsm", "src": "r18_at", "eps": [1, 2, 4, 8], "vec": [2]},
        {"kind": "tr_fgsm_own", "src": "r18_at", "eps": [2, 8], "vec": [2]},
        {"kind": "tr_pgd", "src": "r18_at", "eps": [4, 8], "steps": 5},
    ],
}

SRC = ["r18_std", "r18_at"]


def _settings():
    out = []
    for t in ("std", "at"):
        T = f"r18_{t}"
        base = dict(access="grey", target=T, refs=[T], sources=SRC, n_targets=4)
        out += [
            dict(base, name=f"GB_{t}_all", note="All query and transfer probes."),
            dict(base, name=f"GB_{t}_paperprobe", probe_filter=[rf"^tr_own:{T}:fgsm:e2$"],
                 paper_probe=f"tr_own:{T}:fgsm:e2",
                 note="Only the paper's probe (transfer FGSM w.r.t. the victim's own label, ε = 2/255): "
                      "isolates what calibration / stacking add to the paper's attack."),
            dict(base, name=f"GB_{t}_fgsm", probe_filter=[r"^tr:.*:fgsm"], note="Transfer FGSM probes only."),
            dict(base, name=f"GB_{t}_pgd", probe_filter=[r"^tr:.*:pgd"], note="Transfer PGD probes only."),
            dict(base, name=f"GB_{t}_ownlabel", probe_filter=[r"^tr_own:"],
                 note="Label-free transfer probes (victim's own prediction, as in the paper)."),
            dict(base, name=f"GB_{t}_noise", probe_filter=[r"^gauss:", r"^uniform:"],
                 note="Random-noise neighbourhood only (no surrogate gradients)."),
            dict(base, name=f"WB_{t}_all", access="white", white_box=True,
                 note="White-box: the victim's own FGSM / PGD probes added."),
            dict(base, name=f"WB_{t}_self", access="white", white_box=True, probe_filter=[r"^self"],
                 note="White-box own-gradient probes only."),
            dict(base, name=f"GB_{t}_refs", n_refs=[2, 4, 8, 17], n_targets=3,
                 note="All probes, varying the number of reference models."),
        ]
    return out


SETTINGS = _settings()
SMOKE = dict(n_pop=1500, n_test=500, epochs=2, max_models=5)
REPORT = {}
