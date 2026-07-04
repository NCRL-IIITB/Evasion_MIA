#!/bin/bash
# ============================================================================
# train_both.sh — Train baseline and adversarial victim models sequentially
# ============================================================================
#
# This script:
#   1. Regenerates manifest.csv with a 70/30 patient-level split
#   2. Trains the baseline victim model (DenseNet-121, best-practice recipe)
#   3. Verifies the baseline model was saved correctly
#   4. Trains the adversarially-trained victim model (FGSM defence)
#   5. Verifies the adversarial model was saved correctly
#
# Prerequisites:
#   - mia_env virtual environment exists (run setup.sh first)
#   - NIH Chest X-ray dataset downloaded (run prepare_dataset.py first)
#   - Data_Entry_2017.csv in project root
#
# Usage:
#   bash train_both.sh
#
# Expected runtime on RTX 4090:
#   - Baseline:     ~45-60 min (30 epochs, ~78k member images)
#   - Adversarial:  ~90-120 min (35 epochs, ~3x per-step cost)
#   - Total:        ~2-3 hours
# ============================================================================

set -e  # Exit immediately on any error

# ── Activate virtual environment ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "mia_env" ]; then
    echo "Activating mia_env..."
    source c
else
    echo "ERROR: mia_env not found. Run setup.sh first."
    exit 1
fi

echo ""
echo "============================================================================"
echo "  STEP 1/5: Regenerating manifest with 70/30 patient-level split"
echo "============================================================================"
echo ""

python Victim_Model/prepare_dataset.py --split-ratio 0.7 --split-by-patient

# Verify manifest was created
if [ ! -f "Victim_Model/manifest.csv" ]; then
    echo "ERROR: manifest.csv was not created!"
    exit 1
fi
echo ""
echo "✓ manifest.csv created successfully."

# Verify patient splits JSON was saved
if [ ! -f "Victim_Model/patient_splits.json" ]; then
    echo "WARNING: patient_splits.json was not created."
fi

echo ""
echo "============================================================================"
echo "  STEP 2/5: Training baseline victim model"
echo "============================================================================"
echo ""

python Victim_Model/train_baseline.py

echo ""
echo "============================================================================"
echo "  STEP 3/5: Verifying baseline model"
echo "============================================================================"
echo ""

if [ ! -f "Victim_Model/victim_baseline.pth" ]; then
    echo "ERROR: victim_baseline.pth was not saved! Aborting."
    exit 1
fi

if [ ! -f "Victim_Model/victim_baseline_meta.json" ]; then
    echo "ERROR: victim_baseline_meta.json was not saved! Aborting."
    exit 1
fi

BASELINE_SIZE=$(stat -c%s "Victim_Model/victim_baseline.pth" 2>/dev/null || stat -f%z "Victim_Model/victim_baseline.pth" 2>/dev/null)
echo "✓ victim_baseline.pth saved (${BASELINE_SIZE} bytes)"
echo "✓ victim_baseline_meta.json saved"

echo ""
echo "============================================================================"
echo "  STEP 4/5: Training adversarial victim model (FGSM defence)"
echo "============================================================================"
echo ""

python Victim_Model/train_adversarial.py

echo ""
echo "============================================================================"
echo "  STEP 5/5: Verifying adversarial model"
echo "============================================================================"
echo ""

if [ ! -f "Victim_Model/victim_adversarial.pth" ]; then
    echo "ERROR: victim_adversarial.pth was not saved!"
    exit 1
fi

if [ ! -f "Victim_Model/victim_adversarial_meta.json" ]; then
    echo "ERROR: victim_adversarial_meta.json was not saved!"
    exit 1
fi

ADV_SIZE=$(stat -c%s "Victim_Model/victim_adversarial.pth" 2>/dev/null || stat -f%z "Victim_Model/victim_adversarial.pth" 2>/dev/null)
echo "✓ victim_adversarial.pth saved (${ADV_SIZE} bytes)"
echo "✓ victim_adversarial_meta.json saved"

echo ""
echo "============================================================================"
echo "  ALL MODELS TRAINED SUCCESSFULLY"
echo "============================================================================"
echo ""
echo "Output files:"
echo "  Victim_Model/victim_baseline.pth"
echo "  Victim_Model/victim_baseline_meta.json"
echo "  Victim_Model/logs/victim_baseline_history.csv"
echo "  Victim_Model/logs/victim_baseline_summary.txt"
echo "  Victim_Model/logs/victim_baseline_curves.png"
echo ""
echo "  Victim_Model/victim_adversarial.pth"
echo "  Victim_Model/victim_adversarial_meta.json"
echo "  Victim_Model/logs/victim_adversarial_history.csv"
echo "  Victim_Model/logs/victim_adversarial_summary.txt"
echo "  Victim_Model/logs/victim_adversarial_curves.png"
echo ""
echo "  Victim_Model/patient_splits.json"
echo "  Victim_Model/manifest.csv"
echo ""
echo "Next steps:"
echo "  1. Review training curves in Victim_Model/logs/"
echo "  2. Run MIA: python Membership_Inference_Attack/run_attack.py"
echo "  3. Run FGSM: cd fgsm_attack && python attack_and_compare.py"
echo ""
