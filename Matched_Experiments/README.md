# Matched-control experiment

Answers the concern raised by all three ICDCN reviewers: *the baseline used
augmentation + early stopping while the adversarial model did not, so the
effect of adversarial training cannot be separated from ordinary overfitting.*

Nothing outside this folder is modified. The original attack code
(`Membership_Inference_Attack/`, `Fluctuation_MIA/`, `Victim_Model/api.py`) is
imported unchanged; only its image loading is redirected to a pixel-identical
cache.

## Design

One training script, one data split, paired seeds (42, 43, 44):

|                             | Standard loss | FGSM-AT (ε=0.02, α=0.5) |
|-----------------------------|---------------|-------------------------|
| **aug + early stopping**    | A `std_reg`   | B `fgsm_reg` (+ B2 `fgsm_reg_robustES`) |
| **no aug, 35 epochs, last** | C `std_noreg` | D `fgsm_noreg`          |

Shared by every cell: DenseNet-121 + Dropout(0.3) head, AdamW 1e-4 / wd 1e-4,
batch 64, 35 max epochs, 3-epoch warm-up + cosine (eta_min 1e-6), grad-clip 5,
pos_weight BCE, AMP, the same train/val split, and the same seed (so the same
head initialisation and batch order). Early stopping uses patience 7 on clean
val macro-AUC. B2 is the same run as B, with the checkpoint chosen by
adversarial val AUC (Rice et al., ICML 2020).

Per victim:
- FGSM evasion sweep on all 33,412 non-members, same protocol as the paper's Fig. 5.
- Threshold MIAs (loss, confidence, adversarial loss, white-box fluctuation)
  on the selected model and on every 5-epoch checkpoint → gap-matched analysis.
- The paper's shadow-model MIA (+BCVA) and AFMIA (+BCVA), unchanged
  configuration, plus AUC and TPR at 1 % / 0.1 % FPR.

**Membership:** members are only images the victim trained on. Non-members
come from the patient-disjoint split. The attack seed equals the victim seed.

## Run

```bash
cd /home/iiitb-101/Container_based_Combined_Attack/Evasion_MIA
bash Matched_Experiments/run_matched_experiments.sh --smoke     # pipeline test, ~15 min
nohup bash Matched_Experiments/run_matched_experiments.sh > matched_run.log 2>&1 &
```

- **Optional overrides:** `SEEDS="42"`, `ATTACK_PARALLEL=3`, `RUN_ROBUST_ES=0`,
  `PYTHON=/path/to/python`.
- **Re-running resumes:** completed steps are skipped; failed steps are retried.
- **Progress:** `tail -f Matched_Experiments/results/logs/pipeline.log`.

## Output (`results/`)

```
victims/<cell>_seed<k>/        model.pth, model_meta.json, history.csv, summary.txt, checkpoints/
eval/<run>/                    evasion.json, threshold_mia.json
attacks/shadow_mia/<run>/      results.json, results.txt
attacks/afmia/<run>/           results.json, results.txt
splits/                        attack_split_seed<k>.json (exact pool / eval images)
logs/                          pipeline.log + one log per step
summary/                       REPORT.md, per_run.csv, cell_summary.*, contrasts.*,
                               gap_matched_*.{csv,md,png}, cell_comparison.png
```

`summary/` refreshes every 30 minutes during the run. You can also run
`python Matched_Experiments/aggregate_results.py` at any time.

## Files

| File | Role |
|---|---|
| `matched_common.py` | cache, splits, attacker pool/eval, victim model, FGSM, metrics, I/O patch for the original attack code |
| `build_image_cache.py` | 224×224 cache straight from the NIH zips, checked bit-exact against the original PIL pipeline |
| `train_matched_victim.py` | the single training script for A/B/B2/C/D |
| `evaluate_victim.py` | FGSM evasion + threshold MIAs on checkpoints |
| `run_shadow_attacks.py` | paper's shadow-model MIA / AFMIA (+BCVA) on one victim |
| `aggregate_results.py` | tables, paired-seed contrasts with 95 % CIs, gap-matched fit, figures |
| `run_matched_experiments.sh` | end-to-end driver |
