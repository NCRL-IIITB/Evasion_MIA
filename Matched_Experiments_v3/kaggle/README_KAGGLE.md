# Running the v3 lab on Kaggle

There is one notebook per account, `matched_experiment_v3_lab_kaggle.ipynb`. Each account gets a preset:

| Account | `VICTIMS` | `IMPORT_RUNS` | What it does | Sessions (T4 ×2, rough) |
|---|---|---|---|---|
| 1 — attack-only | `""` | `"std_noreg_seed42 fgsm_noreg_seed42"` | Re-attacks the **old** C and D (v1) with every attack; no training | 2 |
| 2 — new FGSM victims | `"fgsm_rs:4:0.5 fgsm_rs:8:0.5"` | `""` | Trains two FGSM-RS victims in parallel (one per GPU), then attacks both | 3 |
| 3 — optional | `"fgsm_rs:4:0 pgd3:4:0.5"` | `""` | Pure adversarial loss, and 3-step PGD adversarial training | 3–4 |

Account 1's imported C (`v1-std_noreg_seed42`) is the baseline for every adversarial victim of seed 42. Its recipe is identical to v2/v3's standard training, so the merged leaderboard pairs it with all of them.

The regularised victims (A `std_reg_seed42`, B `fgsm_reg_seed42`, B2 `fgsm_reg_robustES_seed42`) can be added to `IMPORT_RUNS` too. They are paired only with each other (same regime).

## Making the old victims available (account 1)

The old victims live in the folders you downloaded: `results 5` (C), `results 3` (D), `results 3_` (A), `results 4` (B, B2). Each contains `victims/<run>/model.pth`, `model_meta.json`, `history.csv` and `checkpoints/`. Use one of these two options:

- **Option A — upload as a private dataset (works from any account).**
  1. kaggle.com → **Create → New Dataset**.
  2. Upload the folders (e.g. `results 5` and `results 3`; ~240 MB each). Keep them private.
  3. In the notebook: **Add Input → Your Datasets →** that dataset.
- **Option B — attach the old notebook output.** Only works on the account that ran it: **Add Input → Your Work → Notebooks →** the old `Matched_Experiments` notebook.

The pipeline searches every attached input for `victims/<name>/model_meta.json` and copies the victim into `results_v3/victims/v1-<name>/`.

## Steps (every account)

1. **Phone-verify the account.** **Push the repository** with all three folders: `Matched_Experiments/`, `Matched_Experiments_v2/`, `Matched_Experiments_v3/`. For a private repository, add the `GITHUB_TOKEN` secret.
2. **Create → New Notebook → File → Import Notebook →** `Matched_Experiments_v3/kaggle/matched_experiment_v3_lab_kaggle.ipynb`.
3. Session options:
   - Accelerator **GPU T4 x2**
   - Internet **On**
4. **Add Input → "NIH Chest X-rays"** (`nih-chest-xrays/data`). On account 1, also attach the old victims (above).
5. Set `LABEL`, `VICTIMS` and `IMPORT_RUNS` to the account's preset.
6. Optional: run `SMOKE = True` once (~45 min) to check the setup. Note that a smoke run only imports victims from *smoke* outputs, so leave `IMPORT_RUNS` empty and set `VICTIMS = "std"` for the smoke test.
7. **Save Version → Save & Run All (Commit).** It stops itself after 11 h.
8. Open the version → **Output → `results_v3/STATUS.md`**:
   - **NOT FINISHED:** **Add Input → Your Work → Notebooks →** this notebook (latest version), then Save Version again.
   - **FAILED:** read `results_v3/logs/<task>.log`.

## Combine all accounts (on your machine)

```bash
python Matched_Experiments_v3/kaggle/merge_kaggle_results.py \
    --out Matched_Experiments_v3/results_kaggle  account1.zip account2.zip account3.zip
```

Then open `Matched_Experiments_v3/results_kaggle/summary/`. It contains:
- `LEADERBOARD.md`: attacks ranked by Δ AUC (D − C) and by strength on D, at image and patient level
- `REPORT.md`: the leaderboard and every paper metric (Table I, Tables II/III, Fig. 6/7, memorization gap, Eq. 7 fluctuation, max-confidence), plus the headline C-vs-D tables, all attack results, evasion detail and gap-matched analysis
- `leaderboard.csv`, `paired.csv`, `attack_results.csv`, `per_run.csv` and the figures

**All accounts must run the same git commit.** The merge refuses to combine different attack splits.

## Output layout (per account)

```
results_v3/
  STATUS.md, kaggle_state.json
  victims/<run>/          trained (fgsm_noreg-…_seed42) or imported (v1-std_noreg_seed42, …)
  eval/<run>/             evasion.json, threshold_mia.json, threshold_scores.npz
  attacks/<run>/          shadows_std/, shadows_at/, shadow_mia_std/, shadow_mia_at/, afmia/, lab/
  splits/, logs/, summary/
```
