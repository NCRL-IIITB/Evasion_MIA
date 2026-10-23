"""
run_shadow_attacks.py
=====================
Run the paper's shadow-model attacks against ONE victim, using the ORIGINAL
attack implementations unchanged:

  --attack shadow_mia : Membership_Inference_Attack/{mia,mia_variance}.py
                        → "Shadow-model MIA" and "Shadow-model MIA + BCVA"
                        (configuration of run_original_mia.py = paper Tables I/II)
  --attack afmia      : Fluctuation_MIA/{mia_fluctuation,mia_fluctuation_variance}.py
                        → "AFMIA" and "AFMIA + BCVA"
                        (configuration of run_fluctuation_attack.py, ε = 0.01)

Configuration (identical to the paper's runs): 8 shadow models cycling
ResNet-18 / MobileNetV3-S / EfficientNet-B0 / DenseNet-121 / ShuffleNetV2,
10,000 images each, 25 epochs, batch 32, Adam lr 1e-3; pool 20k + 20k;
evaluation 5k members + 5k non-members; Gradient Boosting (200, lr 0.05),
Random Forest (200), MLP (256,128), Logistic Regression.

What differs from the original runners
--------------------------------------
* Members are drawn only from images the victim actually trained on.  The
  original drew them from the whole member split; with seed 42 the 15 %
  validation hold-out happened to fall entirely into the unlabelled pool,
  but with other seeds it would be scored as members.
* The attack seed equals the victim seed, so all cells trained with the same
  seed are attacked with the same pool, eval set and shadow initialisations.
* Images come from the 224x224 cache (bit-identical inputs, much faster).
* Besides accuracy / precision / recall / F1 (computed exactly as before from
  attack.predict), each classifier's predict_proba gives AUC and TPR at
  1 % and 0.1 % FPR.
* Resumable (for Kaggle's 12-hour sessions): every trained shadow model, the
  attack dataset and every classifier result are saved in --state_dir as soon
  as they exist; a rerun skips finished work.  The shadow loop replays the
  original _train_shadow_models() exactly (same RNG draws, same calls), it only
  saves/loads each shadow's weights.  The state is deleted on success.

Usage
-----
  python run_shadow_attacks.py --attack afmia --victim_dir results/victims/fgsm_noreg_seed42 \
      --seed 42 --out_dir results/attacks/afmia/fgsm_noreg_seed42
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_common as mc

NUM_SHADOW_MODELS   = 8
SHADOW_DATASET_SIZE = 10_000
SHADOW_EPOCHS       = 25
SHADOW_ARCHS        = ["resnet18", "mobilenet_v3_small", "efficientnet_b0",
                       "densenet121", "shufflenet_v2_x1_0"]
AFMIA_EPSILON       = 0.01
AFMIA_ALPHA         = 0.5
VICTIM_BATCH        = 128

ATTACK_MODELS = [   # identical to run_original_mia.py / run_fluctuation_attack.py
    ("Gradient Boosting",   "gradient_boosting",   dict(n_estimators=200, learning_rate=0.05)),
    ("Random Forest",       "random_forest",       dict(n_estimators=200)),
    ("MLP",                 "mlp",                 dict(hidden_layer_sizes=(256, 128), max_iter=500)),
    ("Logistic Regression", "logistic_regression", dict()),
]


def evaluate_feature_set(obj, feature_label, eval_m, eval_nm, partial: dict, partial_path):
    """Train each attack classifier on obj.attack_dataset and score it.

    Uses the original class's own _train_attack_model() (so each class keeps
    its own feature columns) and attack(..., return_confidence=True) (so the
    victim is queried exactly as in evaluate_attack_model()).  Finished
    classifiers are recorded in ``partial`` and skipped on a rerun.
    """
    from mia import ModelParameters
    X_eval = np.concatenate([eval_m, eval_nm])
    y_true = np.concatenate([np.ones(len(eval_m)), np.zeros(len(eval_nm))])
    rows = []
    for label, model_type, kwargs in ATTACK_MODELS:
        key = f"{feature_label} / {label}"
        if key in partial:
            print(f"  {key:<42s} (already done)", flush=True)
            rows.append(partial[key])
            continue
        t0 = time.time()
        obj.attack_model_parameters = ModelParameters(model_type, **kwargs)
        obj._train_attack_model()
        obj._is_trained = True
        y_pred, proba = obj.attack(X_eval, return_confidence=True)
        score = proba[:, 1] if proba.ndim == 2 else proba
        r = {
            "feature_set":  feature_label,
            "classifier":   label,
            "attack_label": f"{feature_label} / {label}",
            "accuracy":     float(accuracy_score(y_true, y_pred)),
            "precision":    float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":       float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
            "auc":          float(roc_auc_score(y_true, score)),
            "tpr_at_1pct_fpr":   mc.tpr_at_fpr(y_true, score, 0.01),
            "tpr_at_0.1pct_fpr": mc.tpr_at_fpr(y_true, score, 0.001),
            "seconds":      round(time.time() - t0, 1),
        }
        print(f"  {r['attack_label']:<42s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"auc={r['auc']:.4f} tpr@1%={r['tpr_at_1pct_fpr']:.4f}", flush=True)
        rows.append(r)
        partial[key] = r
        mc.write_json(partial_path, partial)
    return rows


def train_shadows_resumable(full, state_dir, afmia):
    """Same as VarianceMIA / FluctuationVarianceMIA._train_shadow_models(),
    plus a saved weights file per shadow so an interrupted run can resume.

    The pool sample of every shadow (rng draws), the victim queries and the
    shadow construction (param.build(random_state + i), whose _build_model()
    re-seeds torch) are replayed in the original order; only fit() is skipped
    for shadows whose weights already exist.
    """
    rng = np.random.RandomState(full.random_state)
    n = len(full.unlabelled_data)
    params = full.shadow_model_parameters
    victim_max_scores = []
    for i in range(full.num_shadow_models):
        train_indices = rng.choice(n, size=full.shadow_model_dataset_size, replace=False)
        train_paths = full.unlabelled_data[train_indices]
        victim_scores = full.victim_model_api.predict(train_paths)
        pseudo_labels = (victim_scores > 0.5).astype(np.float32)
        victim_max_scores.append(np.max(victim_scores, axis=1))

        param = params[i % len(params)] if isinstance(params, list) else params
        shadow = param.build(random_state=full.random_state + i)
        ckpt = os.path.join(state_dir, f"shadow_{i}.pth")
        if os.path.exists(ckpt):
            shadow.model = shadow._build_model()
            shadow.model.load_state_dict(torch.load(ckpt, map_location=shadow.device,
                                                    weights_only=True))
            print(f"  Shadow model {i + 1}/{full.num_shadow_models} loaded "
                  f"({shadow.architecture}) from {ckpt}", flush=True)
        else:
            t0 = time.time()
            shadow.fit(train_paths, pseudo_labels)
            torch.save(shadow.model.state_dict(), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
            print(f"  Shadow model {i + 1}/{full.num_shadow_models} trained "
                  f"({shadow.architecture}, {time.time() - t0:.0f}s) → saved", flush=True)

        full.shadow_models[i] = {"model": shadow, "train_indices": set(train_indices.tolist())}
        if afmia:
            full._shadow_pseudo_labels[i] = {"train_paths": train_paths,
                                             "pseudo_labels": pseudo_labels}
            if full._reference_shadow is None:
                full._reference_shadow = shadow
    full._victim_max_conf_pool = np.concatenate(victim_max_scores)


def main():
    p = argparse.ArgumentParser(description="Paper shadow-model attacks on one victim")
    p.add_argument("--attack", required=True, choices=["shadow_mia", "afmia"])
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=mc.DEFAULT_CACHE_DIR)
    p.add_argument("--splits_dir", default=None, help="where to record the attack split")
    p.add_argument("--state_dir", default=None,
                   help="resume state (default: <out_dir>/_state; removed on success)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    afmia = args.attack == "afmia"
    n_shadow, shadow_size, shadow_epochs = NUM_SHADOW_MODELS, SHADOW_DATASET_SIZE, SHADOW_EPOCHS
    if args.smoke:
        n_shadow, shadow_size, shadow_epochs = 2, 200, 1

    state_dir = args.state_dir or os.path.join(args.out_dir, "_state")
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "README.txt"), "w") as f:
        f.write("Resume state of run_shadow_attacks.py — deleted automatically on success.\n")

    torch.backends.cudnn.benchmark = True
    mc.import_original_attack_modules(afmia=afmia)
    cache = mc.ImageCache(args.cache_dir)
    inference_cls = mc.patch_original_attack_io(cache)
    CachedVictimAPI = mc.make_cached_victim_api(cache, inference_cls)

    from mia import MIA, ModelParameters
    from mia_variance import VarianceMIA
    if afmia:
        import mia as mia_module
        from shadow_models_fluctuation import FluctuationShadowModel
        from mia_fluctuation import FluctuationMIA
        from mia_fluctuation_variance import FluctuationVarianceMIA
        # as in run_fluctuation_attack.py: "pytorch_cnn" builds adversarial shadows
        mia_module._MODEL_REGISTRY["pytorch_cnn"] = FluctuationShadowModel

    # ── Data: corrected membership, seed-specific attack split ────────────────
    pool, eval_m, eval_nm = mc.attack_split_for_seed(args.seed, smoke=args.smoke)
    if args.splits_dir:
        mc.save_attack_split(os.path.join(args.splits_dir, f"attack_split_seed{args.seed}.json"),
                             args.seed, pool, eval_m, eval_nm)
    print(f"[SETUP] attack={args.attack} seed={args.seed} pool={len(pool)} "
          f"eval={len(eval_m)}+{len(eval_nm)}", flush=True)

    model_path = os.path.join(args.victim_dir, "model.pth")
    with open(os.path.join(args.victim_dir, "model_meta.json")) as f:
        meta = json.load(f)
    num_classes = int(meta["num_classes"])
    api = CachedVictimAPI(model_path, num_classes=num_classes, batch_size=VICTIM_BATCH)

    t_start = time.time()
    s_m, s_nm = api.predict(eval_m), api.predict(eval_nm)
    conf_gap = float(s_m.max(axis=1).mean() - s_nm.max(axis=1).mean())
    print(f"[DIAG] max-confidence gap (member − non-member): {conf_gap:+.4f}", flush=True)

    common_kw = dict(victim_model_api=api, unlabelled_data=pool, num_classes=num_classes,
                     num_shadow_models=n_shadow, shadow_model_dataset_size=shadow_size)
    shadow_kw = dict(num_classes=num_classes, epochs=shadow_epochs, batch_size=32, lr=1e-3)
    if afmia:
        shadow_kw.update(epsilon=AFMIA_EPSILON, alpha=AFMIA_ALPHA)
    shadow_params = [ModelParameters("pytorch_cnn", architecture=a, **shadow_kw)
                     for a in SHADOW_ARCHS]

    # ── Shadow phase once (via the BCVA superset class), as the originals do ──
    if afmia:
        full = FluctuationVarianceMIA(shadow_model_parameters=shadow_params,
                                      epsilon=AFMIA_EPSILON, **common_kw)
    else:
        full = VarianceMIA(shadow_model_parameters=shadow_params, **common_kw)
    full.random_state = args.seed
    print("[SHADOW] training shadow models …", flush=True)
    train_shadows_resumable(full, state_dir, afmia)
    dataset_path = os.path.join(state_dir, "attack_dataset.pkl")
    if os.path.exists(dataset_path):
        import pandas as pd
        full.attack_dataset = pd.read_pickle(dataset_path)
        print(f"[SHADOW] attack dataset loaded from {dataset_path}", flush=True)
    else:
        print("[SHADOW] building attack dataset …", flush=True)
        full._prepare_attack_dataset()
        full.attack_dataset.to_pickle(dataset_path + ".tmp")
        os.replace(dataset_path + ".tmp", dataset_path)
    t_shadow = time.time() - t_start

    if afmia:
        base = FluctuationMIA(shadow_model_parameters=shadow_params,
                              epsilon=AFMIA_EPSILON, **common_kw)
        base._shadow_pseudo_labels = full._shadow_pseudo_labels
        base._reference_shadow = full._reference_shadow
        feature_sets = [("AFMIA", base), ("AFMIA + BCVA", full)]
    else:
        base = MIA(shadow_model_parameters=shadow_params, **common_kw)
        feature_sets = [("Shadow-model MIA", base), ("Shadow-model MIA + BCVA", full)]
    base.random_state = args.seed
    base.shadow_models = full.shadow_models
    base.attack_dataset = full.attack_dataset

    partial_path = os.path.join(state_dir, "partial_results.json")
    partial = json.load(open(partial_path)) if os.path.exists(partial_path) else {}
    results = []
    for label, obj in feature_sets:
        print(f"\n[EVAL] {label}", flush=True)
        results.extend(evaluate_feature_set(obj, label, eval_m, eval_nm, partial, partial_path))

    out = {
        "attack": args.attack,
        "victim_dir": args.victim_dir,
        "seed": args.seed,
        "conf_gap": conf_gap,
        "config": {
            "num_shadow_models": n_shadow, "shadow_dataset_size": shadow_size,
            "shadow_epochs": shadow_epochs, "shadow_archs": SHADOW_ARCHS,
            "afmia_epsilon": AFMIA_EPSILON if afmia else None,
            "pool": len(pool), "eval_members": len(eval_m), "eval_nonmembers": len(eval_nm),
            "member_definition": "victim training images only (validation excluded)",
        },
        "runtime_s": {"shadow_phase": round(t_shadow, 1),
                      "total": round(time.time() - t_start, 1)},
        "results": results,
        "victim_meta": {k: v for k, v in meta.items() if k != "label_names"},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    mc.write_json(os.path.join(args.out_dir, "results.json"), out)
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write(f"{args.attack} on {args.victim_dir} (seed {args.seed})\n{'=' * 104}\n")
        f.write(f"{'Attack':<44s}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}"
                f"{'TPR@1%':>9}{'TPR@.1%':>9}\n{'-' * 104}\n")
        for r in results:
            f.write(f"{r['attack_label']:<44s}{r['accuracy']:8.4f}{r['precision']:8.4f}"
                    f"{r['recall']:8.4f}{r['f1']:8.4f}{r['auc']:8.4f}"
                    f"{r['tpr_at_1pct_fpr']:9.4f}{r['tpr_at_0.1pct_fpr']:9.4f}\n")
        f.write(f"\nmax-confidence gap: {conf_gap:+.4f}\nruntime: {out['runtime_s']}\n")
    shutil.rmtree(state_dir, ignore_errors=True)
    print(f"\n[DONE] {args.attack} → {args.out_dir} ({out['runtime_s']['total']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
