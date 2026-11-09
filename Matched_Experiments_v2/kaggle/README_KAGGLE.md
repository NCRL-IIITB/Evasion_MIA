# Running v2 on Kaggle — two accounts

| Account | `CELL` | Victim | Trains for |
|---|---|---|---|
| 1 | `std_noreg` | C — standard training, no augmentation, 35 epochs | ~4 h (or **0 h** with `IMPORT_OLD_VICTIM`) |
| 2 | `fgsm_noreg` | D — FGSM-RS adversarial training, ε = 4/255, α = 0.5, no augmentation, 35 epochs | ~9 h |

For its victim, each account runs:
- training
- evasion (paper grid, pixel grid, PGD)
- the white-box reference MIAs on every checkpoint
- both shadow sets (standard and adversarially trained)
- Shadow-MIA (± BCVA, AT-shadow variant)
- AFMIA (paper features ± BCVA, and AFMIA-v2)

The time estimates are rough and assume a T4.

## Before the first run (once per account)

1. **Phone-verify the account** (Kaggle → Settings → Phone verification). GPU and internet access require it.
2. **Push the repository** with the `Matched_Experiments_v2/` folder. v2 imports `Matched_Experiments/matched_common.py` and `Matched_Experiments/build_image_cache.py`, so push that folder too.
3. If the repository is **private**: notebook editor → **Add-ons → Secrets → Add secret**, named `GITHUB_TOKEN`, with a token that can read the repository.

## Create the notebook (once per account)

1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `Matched_Experiments_v2/kaggle/matched_experiment_v2_kaggle.ipynb`.
2. Right-hand panel → **Session options**:
   - Accelerator: **GPU T4 x2** (not P100)
   - Internet: **On**
3. **Add Input** → search **"NIH Chest X-rays"** (by *National Institutes of Health*, `nih-chest-xrays/data`) → Add.
4. In the second cell set `CELL`:
   - account 1: `"std_noreg"`
   - account 2: `"fgsm_noreg"`
5. Leave `FGSM_VARIANTS = "fgsm_rs:4:0.5"` unless you want a second ε. See *Optional* below.

**Account 1 shortcut — reuse the old C victim (saves ~4 GPU hours).** v2's C uses exactly v1's recipe. If this account still has the old `Matched_Experiments` notebook that produced `results 5` (`std_noreg`):
1. **Add Input → Your Work → Notebooks →** that notebook.
2. Set `IMPORT_OLD_VICTIM = True`.

The pipeline copies `victims/std_noreg_seed42/` (weights, history, checkpoints) and skips training. If the output is on another account, instead upload your local `results 5` folder as a private Kaggle dataset (**Create → New Dataset**) and attach that.

**Test first (recommended):** set `SMOKE = True`, Save & Run All, and wait ~30–45 min. Check that `results_v2_smoke/STATUS.md` says COMPLETE, then set `SMOKE = False`.

## Run

1. **Save Version → Save & Run All (Commit).** It runs in the background, so you can close the browser.
2. The pipeline stops itself after **11 h**, before Kaggle's 12-hour limit, so the output is always saved.
3. Open the finished version → **Output** → `results_v2/STATUS.md`:
   - **COMPLETE:** done.
   - **NOT FINISHED:** **Add Input → Your Work → Notebooks →** this same notebook (its *latest* version), then Save Version again. It resumes:
     - training from its last epoch
     - shadows from their last finished shadow
     - evaluation from its last finished part
   - **FAILED:** read `results_v2/logs/<task>.log`.

Expected sessions:
- **Account 1:** 1 session with `IMPORT_OLD_VICTIM`, 2 without.
- **Account 2:** 2 sessions. The first is mostly training.

**Both accounts must run the same git commit.** The commit is logged in `results_v2/logs/kaggle_pipeline.log`.

## Optional: a second ε on account 2 for free GPU time

While account 2 trains D on one GPU, the second GPU idles. Setting `FGSM_VARIANTS = "fgsm_rs:4:0.5 fgsm_rs:8:0.5"`:
- trains an ε = 8/255 victim in parallel
- attacks both victims

This gives a dose–response point: leakage should grow with ε if adversarial training causes it. The attack phase takes about twice as long (~3 sessions).

## Combine the two accounts (on your own machine)

1. On each account, open the final version → **Output** → **Download**.
2. From the repository root:

   ```bash
   python Matched_Experiments_v2/kaggle/merge_kaggle_results.py \
       --out Matched_Experiments_v2/results_kaggle  baseline.zip fgsm.zip
   ```

3. Open `Matched_Experiments_v2/results_kaggle/summary/REPORT.md`:
   - **Paper-format results:** Table I with robustness gain, Tables II/III with Acc/Prec/Rec/F1 for the 4 feature strategies × 4 classifiers, Fig. 6/7 numbers and figures, memorization gap, Eq. 7 fluctuation, max-confidence.
   - **Headline:** the three attacks, C vs D, with D − C and its paired-bootstrap 95 % CI, and whether the expected direction holds.
   - All feature sets × classifiers, classifier-free scores, evasion detail, gap-matched analysis.

## Output layout

```
results_v2/
  STATUS.md, kaggle_state.json
  victims/<run>/        model.pth, model_meta.json, history.csv (incl. PGD monitor), checkpoints/
  eval/<run>/           evasion.json (paper grid, pixel grid, PGD), threshold_mia.json, threshold_scores.npz
  attacks/<run>/        victim_pool_probs.npy, shadows_std/, shadows_at/ (weights kept),
                        shadow_mia_std/, shadow_mia_at/, afmia/  (results.json/.txt, scores.npz, _cache/)
  splits/               attack_split_seed42.json
  summary/              REPORT.md, per_run.csv, attack_results.csv, paired.csv, *.png
```
