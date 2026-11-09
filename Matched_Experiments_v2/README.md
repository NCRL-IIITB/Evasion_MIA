# Matched experiment v2 — baseline vs FGSM victim, fixed recipe and fixed attacks

v1 (`Matched_Experiments/`) answered the reviewers' confound question. With both
victims unregularised (C `std_noreg`, D `fgsm_noreg`), the confidence-based
MIAs scored the same on both models (≈ 0.70 accuracy), because both memorise
their training set completely.

v2 keeps the matched design and changes two things:

1. **How the FGSM victim is trained.**
   - v1's FGSM training was too weak to change the model's clean behaviour.
   - It also showed signs of catastrophic overfitting.
2. **How AFMIA probes the victim.**
   - v1's black-box AFMIA lost the adversarial-training signal that a white-box probe still sees.
   - That signal is strong in v1's own data: white-box fluctuation AUC was 0.68 on D vs 0.42 on C.

The two victims are run on **two Kaggle accounts** — see [kaggle/README_KAGGLE.md](kaggle/README_KAGGLE.md).

## What changed vs v1

| | v1 | v2 | Why |
|---|---|---|---|
| FGSM victim attack | FGSM, no random start | **FGSM-RS** (random start, 1.25·ε step; Wong et al. 2020) | v1's D showed catastrophic overfitting: FGSM val AUC jumped 0.87 → 0.41 between epochs, and evasion AUC was non-monotonic in ε |
| FGSM victim ε | 0.02 normalised ≈ **1.15/255** | **4/255** (pixel units, per channel) | Too weak to change clean behaviour: val acc 0.920 vs 0.918 |
| α (clean-loss weight) | 0.5 | 0.5 (configurable; 0 = pure AT) | |
| Clamp | ±3 in normalised space | Valid image range | |
| Training monitor | FGSM val AUC | FGSM **and PGD-5** val AUC every epoch | Detects catastrophic overfitting |
| Baseline C | `std_noreg` | **identical recipe** | The old C can be imported instead of retrained |
| Evasion | FGSM at 5 normalised ε | **Paper grid (unchanged)**, plus FGSM at 0.5–8/255 and PGD-10 at 1/2/4/255 | |
| Shadows | Deleted after each attack | **Kept** (resume, re-analysis) | |
| Shadow-MIA | Paper | Paper (+ BCVA), plus a variant with AT shadows | |
| AFMIA | Paper | Paper (+ BCVA), plus **AFMIA-v2** | |
| Metrics | Acc / Prec / Rec / F1 / AUC / TPR | All of those, **plus** acc@median (label-free calibration), per-sample scores, paired-bootstrap CIs for D − C | |
| "Gradient Boosting" | sklearn `GradientBoostingClassifier(200, lr 0.05)` | `HistGradientBoostingClassifier(max_iter 200, lr 0.05)` | Same model family; minutes instead of hours |
| Attack rows | 5k + 5k rows per shadow | 10k pool images × 8 shadows, balanced per shadow (≈ 40k rows) | One shared row set for every feature set; transfer-only sources for AFMIA-v2 |

AFMIA-v2 changes four things compared with the paper's AFMIA:
- **Ensemble transfer:** the probe direction comes from all 8 shadows' gradients, not just shadow 0.
- **Train = test:** shadow rows are probed by transfer from shadows that never saw the image, exactly as the victim is.
- **Several probe sizes:** ε ∈ {1, 2, 4, 8}/255.
- **Logit-space features:** |Δlogit| as well as |Δp|.

It comes in two variants: a **perturbation-only** one (no clean-confidence features, so it isolates the robust-overfitting signal) and one with clean confidence added.

## The three attacks compared (C vs D)

| Attack | Expected | Where |
|---|---|---|
| FGSM evasion | Works on C, not on D | `eval/<run>/evasion.json` |
| Shadow-model MIA ("Original MIA" ± Variance) | Higher on D | `attacks/<run>/shadow_mia_std/` |
| AFMIA ("Fluctuation" ± Variance, plus AFMIA-v2) | Higher on D | `attacks/<run>/afmia/` |

White-box threshold MIAs are computed as upper bounds, on the final model and on every 5-epoch checkpoint:
- loss
- confidence
- adversarial loss (FGSM and PGD)
- fluctuation

## Every metric of the paper is in the report

`summary/REPORT.md` → **"Paper-format results"**:

- **Table I:** clean AUC, then adversarial AUC and flip rate at ε ∈ {0.001, 0.002, 0.005, 0.01, 0.02} (normalised units, clamp ±3, all 33,412 non-members), plus robustness gain.
- **Tables II / III:** for Original MIA, Original MIA + Variance, Fluctuation and Fluctuation + Variance × Gradient Boosting / Random Forest / MLP / Logistic Regression — Acc, Prec, Rec and F1.
- **Fig. 6 and Fig. 7:** figures and their numbers (`paper_fig6_*.png`, `paper_fig7_*.png`).
- **Text metrics:**
  - train acc, val acc and memorization gap
  - epochs trained
  - clean test AUC
  - average max-confidence, members vs non-members
  - Eq. 7 average per-class fluctuation, members vs non-members (paper: 2.67 % vs 7.41 %)

Beyond the paper, the report adds:
- a C-vs-D headline table with paired-bootstrap CIs
- every feature set × classifier with AUC / TPR@1% / TPR@0.1% / acc@0.5 / acc@median / F1
- classifier-free scores
- the evasion detail
- the gap-matched analysis

## Files

| File | Role |
|---|---|
| `v2_common.py` | Imports v1's `matched_common.py` (cache, split, model). Adds pixel-ε FGSM / FGSM-RS / N-FGSM / PGD, shadow architectures and metrics |
| `train_victim.py` | Trains C or D (resumable; deadline-aware) |
| `evaluate_victim.py` | Evasion (paper grid, pixel grid, PGD) and white-box threshold MIAs (resumable) |
| `train_shadows.py` | One shadow set (`std` or `at`), saved shadow by shadow |
| `run_attacks.py` | Shadow-MIA (± BCVA, AT-shadow variant), AFMIA (paper ± BCVA, v2 variants) |
| `aggregate_results.py` | `summary/REPORT.md`, CSVs, figures; paired C-vs-D bootstrap after merging |
| `kaggle/` | Kaggle driver, notebook, merge script, instructions |

Nothing outside this folder was modified, except two additions in `v2_common.py` / `aggregate_results.py` that v3 also uses: a `pgd3` training option, and same-regime pairing.

## Run locally (lab GPU) instead of Kaggle

```bash
cd Evasion_MIA
S=42; R=Matched_Experiments_v2/results_v2
python Matched_Experiments_v2/train_victim.py --training standard --seed $S --out_dir $R/victims/std_noreg_seed$S
python Matched_Experiments_v2/train_victim.py --training fgsm --method fgsm_rs --eps 4 --alpha 0.5 \
       --seed $S --out_dir $R/victims/fgsm_noreg_seed$S
for RUN in std_noreg_seed$S fgsm_noreg_seed$S; do
  python Matched_Experiments_v2/evaluate_victim.py --victim_dir $R/victims/$RUN --seed $S --out_dir $R/eval/$RUN
  for K in std at; do
    python Matched_Experiments_v2/train_shadows.py --kind $K --victim_dir $R/victims/$RUN --seed $S \
           --attack_dir $R/attacks/$RUN --splits_dir $R/splits
  done
  python Matched_Experiments_v2/run_attacks.py --attack shadow_mia --shadow_kind std --victim_dir $R/victims/$RUN --seed $S --attack_dir $R/attacks/$RUN
  python Matched_Experiments_v2/run_attacks.py --attack shadow_mia --shadow_kind at  --victim_dir $R/victims/$RUN --seed $S --attack_dir $R/attacks/$RUN
  python Matched_Experiments_v2/run_attacks.py --attack afmia --victim_dir $R/victims/$RUN --seed $S --attack_dir $R/attacks/$RUN
done
python Matched_Experiments_v2/aggregate_results.py --results_dir $R
```

This uses the local image cache `Matched_Experiments/cache` and `Victim_Model/manifest.csv`, exactly like v1.
