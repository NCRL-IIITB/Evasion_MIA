#!/usr/bin/env bash
# ==============================================================================
# run_pipeline.sh
# ==============================================================================
# End-to-end pipeline:
#   1. Check if victim models exist, train any that are missing
#      - Baseline (standard training, full regularization)
#      - Adversarial ε=0.02 (no augmentation, no early stopping)
#      - Adversarial ε=0.1  (no augmentation, no early stopping)
#   2. Run MIA attack on selected victim model(s)
#
# Usage:
#   bash run_pipeline.sh                                # full pipeline (all 3 models)
#   bash run_pipeline.sh --skip-train                   # skip training, run MIA on all
#   bash run_pipeline.sh --skip-train --victim adversarial_eps002  # MIA on ε=0.02 only
#   bash run_pipeline.sh --skip-train --victim adversarial_eps01   # MIA on ε=0.1 only
#   bash run_pipeline.sh --victim adversarial_eps002 --victim adversarial_eps01  # both adv
#
# --victim options: baseline, adversarial_eps002, adversarial_eps01, both, all (default)
#   You can pass --victim multiple times to select specific models.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VICTIM_DIR="$SCRIPT_DIR/Victim_Model"

# Activate virtualenv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "mia_env/bin/activate" ]]; then
        echo "[SETUP] Activating mia_env ..."
        source mia_env/bin/activate
    else
        echo "[ERROR] mia_env not found and no virtualenv active. Exiting."
        exit 1
    fi
fi

echo "========================================================================"
echo "  Attacks-on-ML Pipeline"
echo "  $(date)"
echo "========================================================================"
echo ""

SKIP_TRAIN=false
VICTIM_TARGETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train) SKIP_TRAIN=true; shift ;;
        --victim)
            if [[ -n "${2:-}" ]]; then
                VICTIM_TARGETS+=("$2"); shift 2
            else
                echo "[ERROR] --victim requires a value"; exit 1
            fi ;;
        *) echo "[WARNING] Unknown argument: $1"; shift ;;
    esac
done

# Default: run all if no --victim flags given
if [[ ${#VICTIM_TARGETS[@]} -eq 0 ]]; then
    VICTIM_TARGETS=("all")
fi

# --------------------------------------------------------------------------
# Step 1: Check and train victim models as needed
# --------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == false ]]; then
    echo "========================================================================"
    echo "  STEP 1: Checking victim models"
    echo "========================================================================"
    echo ""

    # --- Baseline model ---
    if [[ -f "$VICTIM_DIR/victim_baseline.pth" && -f "$VICTIM_DIR/victim_baseline_meta.json" ]]; then
        echo "[✓] Baseline model already exists. Skipping."
    else
        echo "[!] Baseline model NOT found. Training ..."
        python Victim_Model/train_baseline.py
        echo "[✓] Baseline model trained."
    fi
    echo ""

    # --- Adversarial ε=0.02 (no augmentation, no early stopping) ---
    if [[ -f "$VICTIM_DIR/victim_adversarial_eps002_noaug.pth" && -f "$VICTIM_DIR/victim_adversarial_eps002_noaug_meta.json" ]]; then
        echo "[✓] Adversarial (ε=0.02, no-aug) model already exists. Skipping."
    else
        echo "[!] Adversarial (ε=0.02, no-aug) model NOT found. Training ..."
        python Victim_Model/train_adversarial.py \
            --epsilon 0.02 \
            --alpha 0.5 \
            --epochs 35 \
            --no-augmentation \
            --no-early-stopping \
            --tag eps002_noaug
        echo "[✓] Adversarial (ε=0.02, no-aug) model trained."
    fi
    echo ""

    # --- Adversarial ε=0.1 (no augmentation, no early stopping) ---
    if [[ -f "$VICTIM_DIR/victim_adversarial_eps01_noaug.pth" && -f "$VICTIM_DIR/victim_adversarial_eps01_noaug_meta.json" ]]; then
        echo "[✓] Adversarial (ε=0.1, no-aug) model already exists. Skipping."
    else
        echo "[!] Adversarial (ε=0.1, no-aug) model NOT found. Training ..."
        python Victim_Model/train_adversarial.py \
            --epsilon 0.1 \
            --alpha 0.5 \
            --epochs 35 \
            --no-augmentation \
            --no-early-stopping \
            --tag eps01_noaug
        echo "[✓] Adversarial (ε=0.1, no-aug) model trained."
    fi
    echo ""

    echo "========================================================================"
    echo "  All victim models ready."
    echo "========================================================================"
    echo ""
else
    echo "[SKIP] Skipping training (--skip-train flag set)"
    echo ""
fi

# --------------------------------------------------------------------------
# Step 2: Run MIA on selected victim model(s)
# --------------------------------------------------------------------------
echo "========================================================================"
echo "  STEP 2: Running MIA Attacks"
echo "  Targets: ${VICTIM_TARGETS[*]}"
echo "========================================================================"
echo ""

for target in "${VICTIM_TARGETS[@]}"; do
    echo "[MIA] Running attack on: $target"
    python Membership_Inference_Attack/run_attack.py --victim "$target"
    echo ""
done

echo ""
echo "========================================================================"
echo "  PIPELINE COMPLETE"
echo "  $(date)"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  Membership_Inference_Attack/logs/attack_results.txt"
echo ""

