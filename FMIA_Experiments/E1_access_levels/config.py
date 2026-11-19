"""E1 — White-box vs grey-box vs black-box membership inference (CIFAR-10).

Question: how much does each kind of attacker knowledge help fluctuation-based MIA
compared with the existing attacks, on a standard and on an adversarially trained
victim?  Includes the paper's claim that the attacker "need not know the training
procedure" (grey-box with the wrong recipe) and the paper's own black-box recipe
(adversarially trained shadows of unknown architecture).

Access levels (what the attacker has)
  white   victim weights: gradients of the victim itself (self-probes), logits;
          reference models of the same architecture and recipe
  grey    victim architecture (+ recipe, or not: *_recipeX / *_recipeMix); only
          query access (probability vector); transfer probes from its own same-arch
          surrogates
  black   query access only; architecture and recipe unknown: references and
          surrogates are other architectures (VGG-11 / WRN-16-4 / MobileNetV2)
"""
EXPERIMENT = "E1_access_levels"
DESCRIPTION = """
**E1 — access levels.** CIFAR-10, victims ResNet-18 trained with standard training (`r18_std`)
or N-FGSM adversarial training at ε = 8/255 (`r18_at`), on a random half of the 50k training
images each (LiRA protocol: every image is a member of about half of the models).  Grey-box
attackers use ResNet-18 references, black-box attackers use VGG-11 / WRN-16-4 / MobileNetV2
references (`mix_*`).  White-box attackers probe the victim with its own gradients.
"""

DATASETS = {
    "cifar10": dict(n_pop=None, n_test=None, extract_bs=500, evasion_eps=[2, 8], robust_eps=8),
}

TRAIN = dict(epochs=50, batch=256, lr=0.1, wd=5e-4, aug=True)
AT = dict(method="nfgsm", eps=8)
MIX = ["vgg11", "wrn16_4", "mobilenetv2"]

FAMILIES = [
    dict(name="r18_std", dataset="cifar10", arch="resnet18", train={}, n_models=18, n_sources=2),
    dict(name="r18_at", dataset="cifar10", arch="resnet18", train=AT, n_models=18, n_sources=2),
    dict(name="mix_std", dataset="cifar10", arch=MIX, train={}, n_models=11, n_sources=2),
    dict(name="mix_at", dataset="cifar10", arch=MIX, train=AT, n_models=11, n_sources=2),
]

PROBES = {
    "cifar10": [
        {"kind": "hflip"},
        {"kind": "augset"},
        {"kind": "gauss", "sigma": [0.01, 0.03], "k": 8},
        {"kind": "self_fgsm", "eps": [1, 2, 4, 8], "vec": [2]},
        {"kind": "self_pgd", "eps": [8], "steps": 10},
        {"kind": "tr_fgsm", "src": "r18_std", "eps": [1, 2, 4, 8], "vec": [2]},
        {"kind": "tr_fgsm", "src": "r18_at", "eps": [1, 2, 4, 8], "vec": [2]},
        {"kind": "tr_fgsm", "src": "mix_std", "eps": [1, 2, 4, 8], "vec": [2]},
        {"kind": "tr_fgsm", "src": "mix_at", "eps": [1, 2, 4, 8], "vec": [2]},
    ],
}


def _settings():
    out = []
    for t, other in (("std", "at"), ("at", "std")):
        T, O = f"r18_{t}", f"r18_{other}"
        out += [
            dict(name=f"WB_{t}", access="white", target=T, refs=[T], sources=[T], white_box=True,
                 note="Victim weights known: probes along the victim's own gradient; same-recipe references."),
            dict(name=f"GB_{t}", access="grey", target=T, refs=[T], sources=[T],
                 note="Architecture and training recipe known; query access only."),
            dict(name=f"GB_{t}_recipeX", access="grey", target=T, refs=[O], sources=[O],
                 note=f"Architecture known, recipe WRONG: references are `{O}` "
                      "(tests the paper's 'no knowledge of the training procedure' claim)."),
            dict(name=f"GB_{t}_recipeMix", access="grey", target=T, refs=[T, O], sources=[T, O],
                 note="Architecture known, recipe unknown: the attacker hedges with both recipes."),
            dict(name=f"BB_{t}", access="black", target=T, refs=["mix_std", "mix_at"],
                 sources=["mix_std", "mix_at"],
                 note="Architecture and recipe unknown: references and surrogates of other architectures."),
            dict(name=f"BB_{t}_queryonly", access="black", target=T, refs=["mix_std", "mix_at"], sources=[],
                 note="Black-box without surrogate gradients: only noise / augmentation queries."),
            dict(name=f"BB_{t}_paper", access="black", target=T, refs=["mix_at"], sources=["mix_at"],
                 note="The paper's recipe: adversarially trained shadows of an unknown architecture."),
        ]
    return out


SETTINGS = _settings()
for _s in SETTINGS:
    _s["n_targets"] = 4

SMOKE = dict(n_pop=1500, n_test=500, epochs=2, max_models=5)
REPORT = {}
