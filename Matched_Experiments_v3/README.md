# Matched experiment v3 — the experiment lab

A test bench for finding which victim recipe and which black-box attack best show the trend:
- **MIA:** leaks more on the FGSM-trained victim than on the standard one.
- **Evasion:** works on the standard victim, not on the FGSM victim.

It runs many victims through **every** attack (the paper's, v2's and new ones) with the same split, evaluation images and metrics. It then ranks the attacks in a leaderboard.

v3 builds on the other two folders, so push all three:
- **v1** (`Matched_Experiments/`): the image cache, membership split and attacker pool.
- **v2** (`Matched_Experiments_v2/`): victim training, evaluation, shadow training, the paper/v2 attacks and the report with every paper metric.
- **v3** adds `lab_attacks.py`, `aggregate_lab.py` and a Kaggle driver for any mix of trained and imported victims.

---

## 1. Why the regularised results (A, B, B2) are ≈ 50 %

All numbers are from `results 3_`, `results 4`, `results 5` and `results 3` (seed 42).

| Cell | Selected / trained epochs | Train acc | Val acc | Gap | Clean loss gap (NM − M) | **White-box loss AUC** | White-box fluct. AUC | Shadow-MIA acc (GB) | AFMIA acc (GB) | FGSM flip @0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| A `std_reg` | 9 / 16 | 0.876 | 0.877 | −0.1 % | 0.016 | **0.525** | 0.503 | 0.500 | 0.501 | 23.5 % |
| B `fgsm_reg` | 13 / 20 | 0.878 | 0.865 | +1.3 % | 0.020 | **0.527** | 0.502 | 0.500 | 0.510 | 10.6 % |
| B2 `fgsm_reg_robustES` | 7 / 20 | 0.870 | 0.852 | +1.8 % | 0.006 | **0.507** | 0.507 | 0.500 | 0.502 | 9.8 % |
| C `std_noreg` | 35 / 35 | 1.000 | 0.918 | +8.2 % | 0.678 | 0.841 | 0.421 | 0.707 | 0.692 | 17.4 % |
| D `fgsm_noreg` | 35 / 35 | 1.000 | 0.920 | +8.0 % | 0.610 | 0.817 | 0.681 | 0.697 | 0.695 | 6.0 % |

The regularised MIAs are not broken. There is almost nothing to detect:

1. **Early stopping picks epochs 7–13, before memorisation starts.** Train and val accuracy are equal, and members and non-members have the same loss (gap 0.006–0.020, vs 0.6–0.7 without regularisation).
2. **The white-box ceiling is ~0.51–0.53.** An attacker who knows the true labels and the model's exact loss on every image reaches AUC 0.507–0.527. A black-box attack sees less than that: per-image loss is close to the optimal membership signal (Sablayrolles et al., ICML 2019). So 0.50 is the correct answer, and no attack change can create leakage that isn't there.
3. **Augmentation means the model never sees the same pixels twice**, which prevents exact-image memorisation.
4. **The adversarial-training leakage appears late.** In the unregularised D, the white-box fluctuation AUC grows 0.51 → 0.54 → 0.57 → 0.62 → 0.68–0.71 at epochs 5 / 10 / 15 / 20 / 25–35. B stops at epoch 13 and B2 at epoch 7, before this channel opens. That is the robust early stopping of Rice et al. working as a **privacy defence**.
5. **The evasion side works.** Flip rate is 23.5 % for A vs 10.6 % for B and 9.8 % for B2.

**How to report it:** under augmentation + early stopping, neither model leaks (≈ chance, with white-box upper bounds ≤ 0.53), while adversarial training still cuts evasion by more than half. That is a positive mitigation result for the paper. The robustness–privacy trade-off can only appear once the model is trained long enough to memorise, which is the no-regularisation regime this lab works on.

## 2. How the three attacks work, and their ceilings

**Shadow-model MIA ("Original MIA")**
1. The attacker queries the victim on an unlabelled pool (20k members + 20k non-members) to get pseudo-labels.
2. It trains 8 shadow models on 10k-image samples of the pool.
3. It learns, with a classifier, how a shadow's confidence vector differs between its own training images and other images.
4. It applies that classifier to the victim's confidence vectors.

It can only see the victim's *clean* confidence, so its ceiling is the white-box confidence/loss attack. It already sits at that ceiling: 0.837 AUC on C (white-box loss 0.841) and 0.831 on D (0.817). **No change to this attack can make D > C.** Only a victim whose adversarial training widens the clean train/test gap can, which is Song et al.'s mechanism (robust models generalise worse but still fit the training set).

**AFMIA ("Fluctuation")**
1. The shadows are adversarially trained.
2. The features are |p(x) − p(x + ε·sign ∇)| per class, plus max confidence and its class.
3. Shadow rows are probed with each shadow's own gradient; the victim is probed by transfer from shadow 0.

The adversarial-training-specific signal is real:
- **D:** white-box fluctuation AUC 0.68. Members are stable (fluctuation 0.74 vs 1.27).
- **C:** 0.42, the opposite direction. On a standard model, members fluctuate *more*.

v1's black-box AFMIA reached ~0.80 AUC on both, but almost entirely through max confidence, because:
- the single-shadow transfer was weak
- probabilities are saturated at 0/1 for memorised members
- the classifier was trained on white-box features but applied to transferred ones

**FGSM evasion**
- A white-box FGSM; flip rate = the fraction of the 15 labels that change.
- It already shows the expected trend.

## 3. What the lab tests (no-regularisation regime)

### Victim axis — what makes adversarial training leak more

| Recipe (`VICTIMS` entry) | Idea | Cost (T4, rough) |
|---|---|---|
| `std` | Baseline C (same recipe as v1/v2; can be imported instead) | ~4 h |
| v1 `fgsm_noreg` (import) | The paper's recipe (plain FGSM, 1.15/255) — attack-side experiments with no training | 0 h |
| `fgsm_rs:4:0.5` | v2 default: FGSM-RS removes catastrophic overfitting; 4/255 is strong enough to change clean behaviour | ~9 h |
| `fgsm_rs:8:0.5` | Stronger: larger robustness–accuracy trade-off, so a wider clean gap and stronger robust overfitting. A **dose–response** point | ~9 h |
| `fgsm_rs:4:0` | Pure adversarial loss (Madry / Song et al.): the clean loss no longer dominates memorisation | ~9 h |
| `pgd3:4:0.5` | Multi-step adversarial training, closest to Rice et al.'s robust-overfitting setting | ~14 h (2 sessions) |

### Attack axis — all black-box unless marked

| Family | Attack | Idea |
|---|---|---|
| paper | Original MIA, + Variance (BCVA) | As in the paper |
| paper | Fluctuation, + Variance | As in the paper |
| v2 | AFMIA-v2, perturbation only / + confidence | Ensemble transfer, train = test probing, ε ∈ {1,2,4,8}/255, logit-space features |
| v2 | Shadow-MIA with AT shadows | Defence-aware attacker (Song et al.'s threat model) |
| **lab** | **LiRA-cal confidence** | Offline-LiRA-style difficulty calibration (Carlini et al., S&P 2022): the victim's per-class margin compared with models that never trained on the image |
| **lab** | **Robustness-cal** | AFMIA-v2 features minus the same features of models that never trained on the image. Isolates "unusually stable *for this image*", the robust-overfitting signature |
| **lab** | **Query robustness (Square + noise)** | **No gradients and no transfer:** a score-based L∞ Square attack (Andriushchenko et al., ECCV 2020; 60 queries at 4/255) on the loss w.r.t. the victim's own prediction, plus random-noise stability. Measures the victim's own local robustness through its API |
| **lab** | All black-box (combined) | Every block above |
| black-box, no classifier | Fixed-direction scores | e.g. `neg_sq_loss_inc_final` = the victim barely moved under the Square attack. Shows whether a signal exists before a classifier learns it |
| reference | White-box loss, confidence, adversarial loss (FGSM / PGD), fluctuation | Upper bounds (full model access) |

The lab also adds **patient-level** metrics. The split is patient-disjoint, and privacy is about patients: a patient's score is the mean over their evaluation images. Weak per-image signals add up over a patient's images.

## 4. Every metric of the paper is in every report

`summary/REPORT.md` (and the merged report) contains the paper-format section produced by v2's aggregator:
- **Table I:** clean AUC, adversarial AUC and flip rate at ε ∈ {0.001 … 0.02} normalised, plus robustness gain.
- **Tables II / III:** Original MIA, + Variance, Fluctuation, + Variance × GB / RF / MLP / LR — Acc / Prec / Rec / F1.
- **Fig. 6 and Fig. 7:** figures and their numbers.
- **Text metrics:** memorization gap (train / val acc), epochs, clean AUC, Eq. 7 average per-class fluctuation for members / non-members, average max-confidence.

It also contains every feature set × classifier (AUC, TPR@1% / 0.1% FPR, acc@0.5, acc@median, F1), the paired C-vs-D bootstrap, the gap-matched analysis, and the **lab leaderboard**.

## 5. How to pick the best approach

Read `summary/LEADERBOARD.md` (also at the top of `REPORT.md`). For every standard / adversarial pair of the same seed and regime, it shows two rankings:

1. **By trend, Δ AUC = D − C** with a paired-bootstrap 95 % CI, at image and patient level. This is the answer to "does the attack leak more on the FGSM victim?".
2. **By strength on D (AUC on D):** the best attack against the adversarially trained model.

Rules for choosing what goes in the paper:
- **Prefer a trend that holds across settings.** Pick an attack whose Δ AUC CI excludes 0 for more than one adversarial recipe (e.g. ε = 4 and 8). It should also be monotone in ε.
- **Use more seeds for the final claim.** Re-run the final pair with seeds 43 and 44 (`SEEDS = "43 44"`, and train C for those seeds too).
- **Report the full table, not just the winner.** That includes the paper-faithful rows, even when they show no trend. If the confidence-based Original MIA stays equal on C and D, say so. The adversarial-training-specific finding is then *"FGSM training opens a robustness channel that probing attacks detect"*.
- **White-box rows are upper bounds, not attacks.** Never present them as black-box results.

## 6. Running it on Kaggle

See [kaggle/README_KAGGLE.md](kaggle/README_KAGGLE.md). In short, there is one notebook per account (`kaggle/matched_experiment_v3_lab_kaggle.ipynb`) with three presets:
- **Account 1 (attack-only):** imports the old C and D from your `results 5` / `results 3` folders; no training.
- **Account 2:** trains `fgsm_rs:4:0.5` and `fgsm_rs:8:0.5` in parallel.
- **Account 3 (optional):** `fgsm_rs:4:0` and `pgd3:4:0.5`.

To combine them, run `kaggle/merge_kaggle_results.py` on all outputs. The leaderboard pairs every adversarial victim with the C of the same seed.

## Files

| File | Role |
|---|---|
| `lab_attacks.py` | The lab attacks on one victim (needs both shadow sets from v2's `train_shadows.py`); writes `attacks/<run>/lab/` in the same format as v2 |
| `aggregate_lab.py` | v2's full report (paper metrics included) + lab leaderboard + patient-level metrics |
| `kaggle/kaggle_pipeline.py` | Kaggle driver: any list of trained victims (`--victims`) and imported victims (`--import_runs`) |
| `kaggle/matched_experiment_v3_lab_kaggle.ipynb` | The notebook to import on Kaggle |
| `kaggle/merge_kaggle_results.py` | Merge the accounts' outputs and build the combined report |
