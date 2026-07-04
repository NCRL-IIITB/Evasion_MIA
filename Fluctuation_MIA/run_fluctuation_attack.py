"""
run_fluctuation_attack.py
=========================
Fluctuation-Based MIA on the NIH Chest X-ray victim models.

Runs BOTH attacks (Fluctuation + Fluctuation-Variance) against victim models
(baseline + adversarial), producing a comparison table.

The key innovation: instead of training the attack model on raw confidence
scores, we train it on the "fluctuation" — the absolute difference between
the model's output on clean vs adversarial inputs.  This exploits the
"Robust Overfitting" phenomenon in adversarially-trained models.

Pre-conditions
--------------
  Victim_Model/manifest.csv                (run: python Victim_Model/prepare_dataset.py)
  Victim_Model/victim_baseline.pth         (run: python Victim_Model/train_baseline.py)
  Victim_Model/victim_adversarial_*.pth    (run: python Victim_Model/train_adversarial.py)

Usage
-----
  python Fluctuation_MIA/run_fluctuation_attack.py
  python Fluctuation_MIA/run_fluctuation_attack.py --victim baseline
  python Fluctuation_MIA/run_fluctuation_attack.py --victim adversarial_eps002
  python Fluctuation_MIA/run_fluctuation_attack.py --epsilon 0.02
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import pandas as pd

# ── Resolve paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VICTIM_DIR   = os.path.join(PROJECT_ROOT, "Victim_Model")
MIA_DIR      = os.path.join(PROJECT_ROOT, "Membership_Inference_Attack")

sys.path.insert(0, SCRIPT_DIR)   # mia_fluctuation.py, shadow_models_fluctuation.py
sys.path.insert(0, MIA_DIR)      # mia.py, shadow_models.py (base classes)
sys.path.insert(0, VICTIM_DIR)   # api.py

MANIFEST_PATH  = os.path.join(VICTIM_DIR, "manifest.csv")
LOGS_DIR       = os.path.join(SCRIPT_DIR, "logs")
RESULTS_TXT    = os.path.join(LOGS_DIR, "attack_results.txt")
RESULTS_JSON   = os.path.join(LOGS_DIR, "attack_results.json")

# Register FluctuationShadowModel in the model registry so that
# ModelParameters("pytorch_cnn", epsilon=..., alpha=...) builds the right class
import mia as _mia_module
from shadow_models_fluctuation import FluctuationShadowModel
_mia_module._MODEL_REGISTRY["pytorch_cnn"] = FluctuationShadowModel

# ── Attack configuration ───────────────────────────────────────────────────────

NUM_SHADOW_MODELS   = 8
SHADOW_DATASET_SIZE = 10_000   # images per shadow model
NUM_POOL_MEMBERS    = 20_000   # member images in attacker's pool
NUM_POOL_NONMEMBERS = 20_000   # non-member images in attacker's pool
NUM_EVAL_MEMBERS    = 5_000    # held-out evaluation set
NUM_EVAL_NONMEMBERS = 5_000
RANDOM_SEED         = 42

# Victim model variants
VICTIM_VARIANTS = [
    {
        "key":        "baseline",
        "label":      "Baseline (best-practice standard)",
        "model_path": os.path.join(VICTIM_DIR, "victim_baseline.pth"),
        "meta_path":  os.path.join(VICTIM_DIR, "victim_baseline_meta.json"),
    },
    {
        "key":        "adversarial_eps002",
        "label":      "Adversarial (ε=0.02, no augmentation)",
        "model_path": os.path.join(VICTIM_DIR, "victim_adversarial_eps002_noaug.pth"),
        "meta_path":  os.path.join(VICTIM_DIR, "victim_adversarial_eps002_noaug_meta.json"),
    },
    {
        "key":        "adversarial_eps01",
        "label":      "Adversarial (ε=0.1, no augmentation)",
        "model_path": os.path.join(VICTIM_DIR, "victim_adversarial_eps01_noaug.pth"),
        "meta_path":  os.path.join(VICTIM_DIR, "victim_adversarial_eps01_noaug_meta.json"),
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_banner(text: str):
    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: {MANIFEST_PATH} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    member_paths    = df[df["split"] == "member"]["path"].values
    nonmember_paths = df[df["split"] == "nonmember"]["path"].values
    print(f"[DATA] Members:     {len(member_paths)}", flush=True)
    print(f"[DATA] Non-members: {len(nonmember_paths)}", flush=True)
    return member_paths, nonmember_paths


def build_pool_and_eval(member_paths_all, nonmember_paths_all):
    """Return (pool_all, eval_member_paths, eval_nonmember_paths)."""
    rng = np.random.RandomState(RANDOM_SEED)

    max_pool_m  = max(0, len(member_paths_all)    - NUM_EVAL_MEMBERS)
    max_pool_nm = max(0, len(nonmember_paths_all) - NUM_EVAL_NONMEMBERS)
    n_pool_m    = min(NUM_POOL_MEMBERS,    max_pool_m)
    n_pool_nm   = min(NUM_POOL_NONMEMBERS, max_pool_nm)

    member_idx    = rng.permutation(len(member_paths_all))
    nonmember_idx = rng.permutation(len(nonmember_paths_all))

    n_eval_m  = min(NUM_EVAL_MEMBERS,    len(member_paths_all)    - n_pool_m)
    n_eval_nm = min(NUM_EVAL_NONMEMBERS, len(nonmember_paths_all) - n_pool_nm)

    pool_member_paths    = member_paths_all[member_idx[:n_pool_m]]
    pool_nonmember_paths = nonmember_paths_all[nonmember_idx[:n_pool_nm]]
    eval_member_paths    = member_paths_all[member_idx[n_pool_m: n_pool_m + n_eval_m]]
    eval_nonmember_paths = nonmember_paths_all[nonmember_idx[n_pool_nm: n_pool_nm + n_eval_nm]]

    pool_all = np.concatenate([pool_member_paths, pool_nonmember_paths])
    rng.shuffle(pool_all)

    print(
        f"\n[SETUP] Pool size:      {len(pool_all)} "
        f"({n_pool_m} member + {n_pool_nm} nonmember, shuffled)"
    )
    print(f"[SETUP] Eval members:    {len(eval_member_paths)}")
    print(f"[SETUP] Eval nonmembers: {len(eval_nonmember_paths)}")
    sys.stdout.flush()

    return pool_all, eval_member_paths, eval_nonmember_paths


def confidence_gap_diagnostic(api, eval_member_paths, eval_nonmember_paths):
    """Print and return the mean max-confidence gap between members and non-members."""
    t0 = time.time()
    member_scores    = api.predict(np.array(eval_member_paths,    dtype=object))
    nonmember_scores = api.predict(np.array(eval_nonmember_paths, dtype=object))
    print(
        f"  {len(eval_member_paths) + len(eval_nonmember_paths)} eval images "
        f"queried in {time.time() - t0:.1f}s",
        flush=True,
    )

    member_conf    = np.max(member_scores,    axis=1).mean()
    nonmember_conf = np.max(nonmember_scores, axis=1).mean()
    gap            = member_conf - nonmember_conf

    print(f"\n[DIAG] Mean max-confidence on members:     {member_conf:.4f}")
    print(f"[DIAG] Mean max-confidence on non-members: {nonmember_conf:.4f}")
    print(
        f"[DIAG] Confidence gap (member - nonmember): {gap:+.4f}  "
        f"(positive = MIA signal exists)"
    )
    sys.stdout.flush()

    if gap < 0.05:
        print(
            "\n  NOTE: Confidence gap is very small — the standard MIA signal "
            "is weak.  The fluctuation-based attack exploits adversarial "
            "robustness differences instead.",
            flush=True,
        )
    return gap


# ─── Attack model variants to compare ─────────────────────────────────────────

ATTACK_MODELS = [
    ("Gradient Boosting",   "gradient_boosting",   dict(n_estimators=200, learning_rate=0.05)),
    ("Random Forest",       "random_forest",        dict(n_estimators=200)),
    ("MLP",                 "mlp",                  dict(hidden_layer_sizes=(256, 128), max_iter=500)),
    ("Logistic Regression", "logistic_regression",  dict()),
]


# ─── Attack runners ───────────────────────────────────────────────────────────

def run_both_attacks(api, pool_paths, eval_member_paths,
                     eval_nonmember_paths, num_classes: int,
                     epsilon: float) -> tuple[list[dict], list[dict]]:
    """Train shadow models ONCE, build datasets, evaluate both attacks.

    Shadow models are adversarially trained (FluctuationShadowModel) and
    shared between the base Fluctuation attack and the Variance-enhanced
    Fluctuation attack.

    Returns
    -------
    (fluctuation_results, variance_results) : tuple of list[dict]
    """
    from mia import ModelParameters
    from mia_fluctuation import FluctuationMIA
    from mia_fluctuation_variance import FluctuationVarianceMIA
    from shadow_models_fluctuation import FluctuationShadowModel

    # 5 diverse architectures — cycle across the 8 shadow models
    shadow_params = [
        ModelParameters(
            "pytorch_cnn", architecture=arch, num_classes=num_classes,
            epochs=25, batch_size=32, lr=1e-3,
            epsilon=epsilon, alpha=0.5,
        )
        for arch in [
            "resnet18",
            "mobilenet_v3_small",
            "efficientnet_b0",
            "densenet121",
            "shufflenet_v2_x1_0",
        ]
    ]

    # ── Step 1: Train shadow models ONCE (via FluctuationVarianceMIA) ─────────
    # FluctuationVarianceMIA._train_shadow_models() also caches the victim's
    # CLEAN max-confidence pool needed for the variance feature.
    print("\n[SHADOW] Training adversarial shadow models (shared by both attacks) …",
          flush=True)
    fvmia = FluctuationVarianceMIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
        epsilon=epsilon,
    )
    print("[SHADOW] Step 1/2: Training shadow models …", flush=True)
    fvmia._train_shadow_models()
    print(f"[SHADOW] ✓ {len(fvmia.shadow_models)} shadow models trained.\n",
          flush=True)

    # ── Step 2: Build fluctuation+variance dataset ONCE (superset) ────────────
    print("[DATASET] Building shared fluctuation+variance attack dataset …",
          flush=True)
    fvmia._prepare_attack_dataset()

    shared_dataset = fvmia.attack_dataset
    print(f"\n  Shared dataset: {len(shared_dataset)} rows × "
          f"{len(shared_dataset.columns)} cols "
          f"({list(shared_dataset.columns)})")
    print("  Sample (first 3 rows):")
    print(shared_dataset.head(3).to_string(index=False))

    # ── Step 3a: Evaluate FLUCTUATION attacks (fluct features only) ───────────
    print("\n[EVAL] Evaluating FLUCTUATION attack models (fluct + max_conf) …",
          flush=True)
    fmia = FluctuationMIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
        epsilon=epsilon,
    )
    fmia.shadow_models          = fvmia.shadow_models
    fmia._shadow_pseudo_labels  = fvmia._shadow_pseudo_labels
    fmia._reference_shadow      = fvmia._reference_shadow
    fmia.attack_dataset         = shared_dataset

    fluct_results = []
    for label, model_type, kwargs in ATTACK_MODELS:
        params  = ModelParameters(model_type, **kwargs)
        metrics = fmia.evaluate_attack_model(params, eval_member_paths, eval_nonmember_paths)
        metrics["attack_label"] = f"Fluctuation / {label}"
        _print_metrics(f"Fluctuation / {label}", metrics)
        fluct_results.append(metrics)

    # ── Step 3b: Evaluate FLUCTUATION+VARIANCE attacks ────────────────────────
    print("\n[EVAL] Evaluating FLUCTUATION+VARIANCE attack models …", flush=True)
    variance_results = []
    for label, model_type, kwargs in ATTACK_MODELS:
        params  = ModelParameters(model_type, **kwargs)
        metrics = fvmia.evaluate_attack_model(params, eval_member_paths, eval_nonmember_paths)
        metrics["attack_label"] = f"Fluct+Variance / {label}"
        _print_metrics(f"Fluct+Variance / {label}", metrics)
        variance_results.append(metrics)

    return fluct_results, variance_results


def _print_metrics(label: str, metrics: dict):
    print(f"\n  {label} Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    sys.stdout.flush()


# ─── Per-victim experiment ─────────────────────────────────────────────────────

def run_experiments_for_victim(variant: dict, pool_all, eval_member_paths,
                               eval_nonmember_paths,
                               epsilon: float) -> list[dict]:
    """Run fluctuation + variance attacks against one victim model."""
    from api import VictimAPI

    model_path = variant["model_path"]
    meta_path  = variant["meta_path"]
    label      = variant["label"]

    if not os.path.exists(model_path):
        print(
            f"\n  SKIPPING {label}: {model_path} not found. "
            f"Run train_baseline.py or train_adversarial.py first.",
            flush=True,
        )
        return []

    with open(meta_path, "r") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])

    print_banner(f"VICTIM: {label.upper()}")
    print(f"  Architecture:    {meta.get('architecture', '?')}")
    print(f"  train_acc:       {meta.get('final_train_acc', 0):.4f}")
    print(f"  val_acc:         {meta.get('final_val_acc', 0):.4f}")
    print(f"  val_AUC:         {meta.get('final_val_auc', 0):.4f}")
    print(f"  Memorization gap:{meta.get('memorization_gap', 0) * 100:+.2f}%")
    print(f"  Dropout:         {meta.get('dropout', False)}")
    print(f"  Weight decay:    {meta.get('weight_decay', 0.0)}")
    print(f"  FGSM epsilon:    {epsilon}")
    sys.stdout.flush()

    victim_meta_snapshot = {
        k: v for k, v in meta.items()
        if k not in ("imagenet_mean", "imagenet_std", "label_names")
    }

    api = VictimAPI(model_path, num_classes=num_classes, batch_size=32)
    print(f"\n  Inference device: {api.device}", flush=True)

    print("\n  Pre-computing confidence-gap diagnostic …", flush=True)
    gap = confidence_gap_diagnostic(api, eval_member_paths, eval_nonmember_paths)

    # ── Run both attacks with SHARED shadow models ────────────────────────────
    print_banner(f"[{label}] FLUCTUATION + VARIANCE ATTACKS")
    t0 = time.time()
    rows_fluct, rows_variance = run_both_attacks(
        api, pool_all, eval_member_paths, eval_nonmember_paths,
        num_classes, epsilon
    )

    results = []
    for r in rows_fluct:
        r["victim_label"] = label
        r["conf_gap"]     = gap
        r["mia_type"]     = "Fluctuation"
        r["victim_meta"]  = victim_meta_snapshot
    results.extend(rows_fluct)

    for r in rows_variance:
        r["victim_label"] = label
        r["conf_gap"]     = gap
        r["mia_type"]     = "Fluct+Variance"
        r["victim_meta"]  = victim_meta_snapshot
    results.extend(rows_variance)

    print(f"\n  Total attack runtime for {label}: {time.time() - t0:.1f}s",
          flush=True)

    return results


def _save_results(all_results: list, total_time: float):
    """Write full results to attack_results.txt and attack_results.json."""
    os.makedirs(LOGS_DIR, exist_ok=True)

    VL = 34
    AL = 38

    with open(RESULTS_TXT, "w") as f:
        f.write("NIH Chest X-ray — Fluctuation MIA Results\n")
        f.write("=" * 90 + "\n\n")
        f.write(
            f"  {'Victim Model':<{VL}}  {'Attack':<{AL}}  "
            f"{'Gap':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}\n"
        )
        f.write("  " + "-" * 88 + "\n")
        for r in all_results:
            f.write(
                f"  {r['victim_label']:<{VL}}  {r['attack_label']:<{AL}}  "
                f"{r['conf_gap']:+7.4f}  {r['accuracy']:7.4f}  "
                f"{r['precision']:7.4f}  {r['recall']:7.4f}  {r['f1']:7.4f}\n"
            )
        f.write(f"\nRandom baseline: 0.5000\n")
        f.write(f"Total runtime:   {total_time:.1f}s\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("Victim Model Details\n")
        f.write("=" * 90 + "\n")
        seen = set()
        for r in all_results:
            vl = r["victim_label"]
            if vl not in seen:
                seen.add(vl)
                f.write(f"\n  {vl}\n")
                vm = r.get("victim_meta", {})
                for k, v in vm.items():
                    f.write(f"    {k:<25s}: {v}\n")

    # JSON results (without non-serializable victim_meta)
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k != "victim_meta"}
        json_results.append(jr)

    with open(RESULTS_JSON, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n  Results (TXT):  {RESULTS_TXT}", flush=True)
    print(f"  Results (JSON): {RESULTS_JSON}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Fluctuation-Based MIA against victim models"
    )
    parser.add_argument(
        "--victim", type=str, default="both",
        choices=["both", "all", "baseline", "adversarial_eps002", "adversarial_eps01"],
        help="Which victim model(s) to attack. "
             "'both'=baseline+eps002, 'all'=all three. Default: both",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="FGSM perturbation magnitude for adversarial generation. Default: 0.01",
    )
    args = parser.parse_args()

    print_banner("Fluctuation-Based MIA — NIH Chest X-ray")
    print(f"\n  FGSM epsilon: {args.epsilon}", flush=True)

    # 1. Load manifest
    print("\n[SETUP] Loading manifest …", flush=True)
    member_paths_all, nonmember_paths_all = load_manifest()

    # 2. Build shared pool + eval split
    pool_all, eval_member_paths, eval_nonmember_paths = build_pool_and_eval(
        member_paths_all, nonmember_paths_all
    )

    # 3. Determine which victims to run
    if args.victim == "all":
        variants = VICTIM_VARIANTS
    elif args.victim == "both":
        variants = [v for v in VICTIM_VARIANTS
                     if v["key"] in ("baseline", "adversarial_eps002")]
    else:
        variants = [v for v in VICTIM_VARIANTS if v["key"] == args.victim]

    # 4. Run experiments
    grand_start = time.time()
    all_results = []

    for variant in variants:
        results = run_experiments_for_victim(
            variant, pool_all, eval_member_paths, eval_nonmember_paths,
            args.epsilon
        )
        all_results.extend(results)
        if all_results:
            _save_results(all_results, time.time() - grand_start)

    total_time = time.time() - grand_start

    if not all_results:
        print("\nNo results — did you train the victim models first?", flush=True)
        return

    # 5. Final comparison table
    print_banner("FINAL COMPARISON TABLE")

    VL = 34
    AL = 38

    header = (
        f"\n  {'Victim Model':<{VL}}  {'Attack':<{AL}}  "
        f"{'Gap':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}"
    )
    print(header)
    print("  " + "-" * 88)

    for r in all_results:
        print(
            f"  {r['victim_label']:<{VL}}  {r['attack_label']:<{AL}}  "
            f"{r['conf_gap']:+7.4f}  {r['accuracy']:7.4f}  "
            f"{r['precision']:7.4f}  {r['recall']:7.4f}  {r['f1']:7.4f}"
        )

    print(f"\n  Random baseline: 0.5000")
    print(f"  Total runtime:   {total_time:.1f}s")
    sys.stdout.flush()

    # 6. Final save
    _save_results(all_results, total_time)


if __name__ == "__main__":
    main()
