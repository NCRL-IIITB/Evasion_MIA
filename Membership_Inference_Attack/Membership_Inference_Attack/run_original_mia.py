"""
run_original_mia.py
===================
Run the standard (confidence-based) Membership Inference Attack against the
same two victim models that the Fluctuation MIA was evaluated on:

  1. victim_baseline.pth              (standard training)
  2. victim_adversarial_eps002_noaug.pth  (FGSM ε=0.02, no augmentation, no early stopping)

For each victim, both Baseline (15-dim confidence) and Variance (16-dim
confidence + bootstrapped variance) attacks are run with four classifiers:
Gradient Boosting, Random Forest, MLP, and Logistic Regression.

Results are saved as separate per-victim text files:
  logs/ORIGINAL_MIA_baseline_RESULTS.txt
  logs/ORIGINAL_MIA_adversarial_RESULTS.txt

Usage
-----
  python Membership_Inference_Attack/run_original_mia.py
  python Membership_Inference_Attack/run_original_mia.py --victim baseline
  python Membership_Inference_Attack/run_original_mia.py --victim adversarial
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

sys.path.insert(0, SCRIPT_DIR)   # mia.py, mia_variance.py, shadow_models.py
sys.path.insert(0, VICTIM_DIR)   # api.py

MANIFEST_PATH = os.path.join(VICTIM_DIR, "manifest.csv")
LOGS_DIR      = os.path.join(SCRIPT_DIR, "logs")

# ── Victim model definitions ─────────────────────────────────────────────────

VICTIM_VARIANTS = {
    "baseline": {
        "label":       "Baseline (best-practice standard)",
        "model_path":  os.path.join(VICTIM_DIR, "victim_baseline.pth"),
        "meta_path":   os.path.join(VICTIM_DIR, "victim_baseline_meta.json"),
        "output_file": os.path.join(LOGS_DIR, "ORIGINAL_MIA_baseline_RESULTS.txt"),
    },
    "adversarial": {
        "label":       "Adversarial (ε=0.02, no augmentation)",
        "model_path":  os.path.join(VICTIM_DIR, "victim_adversarial_eps002_noaug.pth"),
        "meta_path":   os.path.join(VICTIM_DIR, "victim_adversarial_eps002_noaug_meta.json"),
        "output_file": os.path.join(LOGS_DIR, "ORIGINAL_MIA_adversarial_RESULTS.txt"),
    },
}

# ── Attack configuration ─────────────────────────────────────────────────────

NUM_SHADOW_MODELS   = 8
SHADOW_DATASET_SIZE = 10_000
NUM_POOL_MEMBERS    = 20_000
NUM_POOL_NONMEMBERS = 20_000
NUM_EVAL_MEMBERS    = 5_000
NUM_EVAL_NONMEMBERS = 5_000
RANDOM_SEED         = 42

ATTACK_MODELS = [
    ("Gradient Boosting",   "gradient_boosting",   dict(n_estimators=200, learning_rate=0.05)),
    ("Random Forest",       "random_forest",        dict(n_estimators=200)),
    ("MLP",                 "mlp",                  dict(hidden_layer_sizes=(256, 128), max_iter=500)),
    ("Logistic Regression", "logistic_regression",  dict()),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Build the attacker's unlabelled pool and the evaluation split."""
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
    """Compute and print the mean max-confidence gap between members and non-members."""
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
    print(f"[DIAG] Confidence gap (member - nonmember): {gap:+.4f}")
    sys.stdout.flush()
    return gap


# ── Attack runner ─────────────────────────────────────────────────────────────

def run_attacks_for_victim(api, pool_paths, eval_member_paths,
                           eval_nonmember_paths, num_classes):
    """Train shadow models once, then run Baseline and Variance attacks.

    Returns a list of result dicts, each with keys:
      attack_label, accuracy, precision, recall, f1
    """
    from mia import MIA, ModelParameters
    from mia_variance import VarianceMIA

    shadow_params = [
        ModelParameters(
            "pytorch_cnn", architecture=arch, num_classes=num_classes,
            epochs=25, batch_size=32, lr=1e-3,
        )
        for arch in [
            "resnet18",
            "mobilenet_v3_small",
            "efficientnet_b0",
            "densenet121",
            "shufflenet_v2_x1_0",
        ]
    ]

    # Step 1: Train shadow models once via VarianceMIA (superset)
    print("\n[SHADOW] Training shadow models …", flush=True)
    vmia = VarianceMIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
    )
    vmia._train_shadow_models()
    print(f"[SHADOW] {len(vmia.shadow_models)} shadow models trained.\n", flush=True)

    # Step 2: Build 16-dim attack dataset once (superset of baseline's 15-dim)
    print("[DATASET] Building shared attack dataset (conf + variance) …", flush=True)
    vmia._prepare_attack_dataset()
    shared_dataset = vmia.attack_dataset
    print(f"  Shared dataset: {len(shared_dataset)} rows × {len(shared_dataset.columns)} cols")

    # Step 3a: Baseline attacks (15-dim, class_0…class_14 only)
    print("\n[EVAL] Evaluating BASELINE attack models (15-dim confidence) …", flush=True)
    mia = MIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
    )
    mia.shadow_models  = vmia.shadow_models
    mia.attack_dataset = shared_dataset

    results = []
    for label, model_type, kwargs in ATTACK_MODELS:
        params  = ModelParameters(model_type, **kwargs)
        metrics = mia.evaluate_attack_model(params, eval_member_paths, eval_nonmember_paths)
        metrics["attack_label"] = f"Baseline / {label}"
        print(f"  {metrics['attack_label']:40s}  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
        results.append(metrics)

    # Step 3b: Variance attacks (16-dim, conf + variance)
    print("\n[EVAL] Evaluating VARIANCE attack models (16-dim conf+var) …", flush=True)
    for label, model_type, kwargs in ATTACK_MODELS:
        params  = ModelParameters(model_type, **kwargs)
        metrics = vmia.evaluate_attack_model(params, eval_member_paths, eval_nonmember_paths)
        metrics["attack_label"] = f"Variance / {label}"
        print(f"  {metrics['attack_label']:40s}  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
        results.append(metrics)

    return results


def save_results_txt(output_path, victim_label, victim_meta, gap, results, runtime):
    """Write results for a single victim to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    VL = 40
    AL = 40

    with open(output_path, "w") as f:
        f.write("NIH Chest X-ray — Original MIA Results\n")
        f.write("=" * 90 + "\n\n")
        f.write(
            f"  {'Victim Model':<{VL}}  {'Attack':<{AL}}  "
            f"{'Gap':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}\n"
        )
        f.write("  " + "-" * 88 + "\n")
        for r in results:
            f.write(
                f"  {victim_label:<{VL}}  {r['attack_label']:<{AL}}  "
                f"{gap:+7.4f}  {r['accuracy']:7.4f}  "
                f"{r['precision']:7.4f}  {r['recall']:7.4f}  {r['f1']:7.4f}\n"
            )
        f.write(f"\nRandom baseline: 0.5000\n")
        f.write(f"Total runtime:   {runtime:.1f}s\n")

        # Victim metadata block
        f.write("\n" + "=" * 90 + "\n")
        f.write("Victim Model Details\n")
        f.write("=" * 90 + "\n")
        f.write(f"\n  {victim_label}\n")
        for k, v in victim_meta.items():
            f.write(f"    {k:<25s}: {v}\n")

    print(f"\n  Results saved to: {output_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run original (confidence-based) MIA against baseline and adversarial victims"
    )
    parser.add_argument(
        "--victim", type=str, default="both",
        choices=["both", "baseline", "adversarial"],
        help="Which victim model(s) to attack. Default: both",
    )
    args = parser.parse_args()

    print_banner("Original (Confidence-Based) MIA — NIH Chest X-ray")

    # 1. Load manifest
    print("\n[SETUP] Loading manifest …", flush=True)
    member_paths_all, nonmember_paths_all = load_manifest()

    # 2. Build shared pool and eval split
    pool_all, eval_member_paths, eval_nonmember_paths = build_pool_and_eval(
        member_paths_all, nonmember_paths_all
    )

    # 3. Determine which victims to attack
    if args.victim == "both":
        keys = ["baseline", "adversarial"]
    else:
        keys = [args.victim]

    # 4. Run attacks per victim
    for key in keys:
        variant = VICTIM_VARIANTS[key]
        model_path  = variant["model_path"]
        meta_path   = variant["meta_path"]
        label       = variant["label"]
        output_file = variant["output_file"]

        if not os.path.exists(model_path):
            print(f"\n  SKIPPING {label}: {model_path} not found.", flush=True)
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        num_classes = int(meta["num_classes"])

        print_banner(f"VICTIM: {label.upper()}")
        print(f"  Architecture:     {meta.get('architecture', '?')}")
        print(f"  Training type:    {meta.get('training_type', '?')}")
        print(f"  train_acc:        {meta.get('final_train_acc', 0):.4f}")
        print(f"  val_acc:          {meta.get('final_val_acc', 0):.4f}")
        print(f"  val_AUC:          {meta.get('final_val_auc', 0):.4f}")
        print(f"  Memorization gap: {meta.get('memorization_gap', 0) * 100:+.2f}%")
        sys.stdout.flush()

        # Metadata snapshot (exclude bulky fields)
        victim_meta_snapshot = {
            k: v for k, v in meta.items()
            if k not in ("imagenet_mean", "imagenet_std", "label_names")
        }

        # Load victim API
        from api import VictimAPI
        api = VictimAPI(model_path, num_classes=num_classes, batch_size=32)
        print(f"\n  Inference device: {api.device}", flush=True)

        # Confidence gap diagnostic
        print("\n  Computing confidence-gap diagnostic …", flush=True)
        gap = confidence_gap_diagnostic(api, eval_member_paths, eval_nonmember_paths)

        # Run attacks
        t0 = time.time()
        results = run_attacks_for_victim(
            api, pool_all, eval_member_paths, eval_nonmember_paths, num_classes
        )
        runtime = time.time() - t0

        # Save per-victim results
        save_results_txt(output_file, label, victim_meta_snapshot, gap, results, runtime)

        # Print summary
        print_banner(f"RESULTS FOR {label.upper()}")
        for r in results:
            print(
                f"  {r['attack_label']:40s}  "
                f"acc={r['accuracy']:.4f}  prec={r['precision']:.4f}  "
                f"rec={r['recall']:.4f}  f1={r['f1']:.4f}"
            )
        print(f"\n  Runtime: {runtime:.1f}s")
        sys.stdout.flush()

    print_banner("ALL DONE")


if __name__ == "__main__":
    main()
