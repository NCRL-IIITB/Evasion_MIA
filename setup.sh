#!/bin/bash

# Setup script for MIA on NIH Chest X-ray pipeline
# Targets Linux with NVIDIA GPUs

set -e # Exit immediately if a command exits with a non-zero status

echo "=========================================================="
echo " Setting up MIA Pipeline Environment"
echo "=========================================================="

# 1. Create and activate virtual environment
echo "[1/3] Creating virtual environment 'mia_env'..."
python3 -m venv mia_env
source mia_env/bin/activate

# 2. Upgrade pip
echo "[2/3] Upgrading pip..."
pip install --upgrade pip

# 3. Install core dependencies (including PyTorch with cu126)
echo "[3/3] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "=========================================================="
echo " Setup Complete!"
echo "=========================================================="
echo ""
echo "To begin, activate the environment:"
echo "    source mia_env/bin/activate"
echo ""
echo "Execution Order:"
echo "  1. Download data & build manifest:"
echo "     python Victim_Model/prepare_dataset.py"
echo ""
echo "  2. Train victim model (Overfitted):"
echo "     python Victim_Model/train_victim.py --mode overfit"
echo ""
echo "  3. Train victim model (Regularized):"
echo "     python Victim_Model/train_victim.py --mode regularized"
echo ""
echo "  4. Run Attacks:"
echo "     python Membership_Inference_Attack/run_attack.py"
