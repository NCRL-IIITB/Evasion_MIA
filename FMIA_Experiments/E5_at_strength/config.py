"""E5 — Robustness–privacy trade-off: adversarial-training strength and method (CIFAR-10, ResNet-18).

Question: as adversarial training gets stronger (evasion gets harder), does
membership leakage through fluctuation grow while confidence-based leakage
shrinks — and does CF-MIA stay the strongest attack along the whole curve?
This is the paper's central claim, measured as a dose–response curve.

The transfer surrogates are always STANDARD models (`std` family sources), i.e.
the attacker never needs to know the defence.  Each defended family is attacked
with references of its own recipe (GB, recipe known) and with standard references
(GB_recipeX, recipe unknown).
"""
EXPERIMENT = "E5_at_strength"
DESCRIPTION = """
**E5 — adversarial-training strength.** CIFAR-10, ResNet-18.  Standard training vs N-FGSM
adversarial training at ε ∈ {2, 4, 8, 16}/255, PGD-5 at 8/255 and TRADES (β = 6, 5 steps) at
8/255.  Transfer probes come from standard surrogates only (defence-agnostic attacker).
"""

DATASETS = {
    "cifar10": dict(n_pop=None, n_test=None, extract_bs=500, evasion_eps=[2, 8], robust_eps=8),
}

TRAIN = dict(epochs=50, batch=256, lr=0.1, wd=5e-4, aug=True)

FAMILIES = [
    dict(name="std", dataset="cifar10", arch="resnet18", train={}, n_models=10, n_sources=2),
    dict(name="nfgsm2", dataset="cifar10", arch="resnet18", train=dict(method="nfgsm", eps=2), n_models=8),
    dict(name="nfgsm4", dataset="cifar10", arch="resnet18", train=dict(method="nfgsm", eps=4), n_models=8),
    dict(name="nfgsm8", dataset="cifar10", arch="resnet18", train=dict(method="nfgsm", eps=8), n_models=8),
    dict(name="nfgsm16", dataset="cifar10", arch="resnet18", train=dict(method="nfgsm", eps=16), n_models=8),
    # multi-step methods cost ~3x more per model: fewer models, trained after the N-FGSM curve
    dict(name="pgd8", dataset="cifar10", arch="resnet18", train=dict(method="pgd", eps=8, steps=5),
         n_models=6, priority=1),
    dict(name="trades8", dataset="cifar10", arch="resnet18",
         train=dict(method="trades", eps=8, steps=5, beta=6.0), n_models=6, priority=1),
]

PROBES = {
    "cifar10": [
        {"kind": "hflip"},
        {"kind": "augset"},
        {"kind": "gauss", "sigma": [0.01, 0.03], "k": 8},
        {"kind": "self_fgsm", "eps": [2, 8], "vec": [2]},
        {"kind": "self_pgd", "eps": [8], "steps": 10},
        {"kind": "tr_fgsm", "src": "std", "eps": [1, 2, 4, 8, 16], "vec": [2]},
        {"kind": "tr_fgsm_own", "src": "std", "eps": [2], "vec": [2]},
        {"kind": "tr_pgd", "src": "std", "eps": [8], "steps": 5},
    ],
}


def _settings():
    out = []
    for f in FAMILIES:
        F = f["name"]
        out.append(dict(name=f"GB_{F}", access="grey", target=F, refs=[F], sources=["std"],
                        note="Recipe known: references trained like the victim."))
        out.append(dict(name=f"WB_{F}", access="white", target=F, refs=[F], sources=["std"], white_box=True))
        if F != "std":
            out.append(dict(name=f"GB_{F}_recipeX", access="grey", target=F, refs=["std"], sources=["std"],
                            note="Recipe unknown: standard references only."))
    for s in out:
        s["n_targets"] = 3
    return out


SETTINGS = _settings()
SMOKE = dict(n_pop=1500, n_test=500, epochs=2, max_models=5)
REPORT = {"tradeoff_settings": [f"GB_{f['name']}" for f in FAMILIES]}
