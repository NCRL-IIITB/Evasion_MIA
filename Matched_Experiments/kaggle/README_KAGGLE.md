# Running the matched experiment on Kaggle (free tier)

One Kaggle account runs one victim cell. Four accounts cover the 2×2 design:

| Account | `CELL` | What it trains and attacks |
|---|---|---|
| 1 | `std_reg`    | A: standard loss, augmentation + early stopping |
| 2 | `fgsm_reg`   | B: FGSM-AT, augmentation + early stopping, **plus B2** (same run, robust early stopping) |
| 3 | `std_noreg`  | C: standard loss, no augmentation, 35 epochs |
| 4 | `fgsm_noreg` | D: FGSM-AT, no augmentation, 35 epochs |

For its victim, each account runs training, FGSM evasion, threshold MIAs on all
checkpoints, the shadow-model MIA (+BCVA) and AFMIA (+BCVA).

## Before the first run (once per account)

1. **Phone-verify the account** (Kaggle → Settings → Phone verification).
   GPU and internet access require it.
2. **Push the latest code to GitHub** (from the lab machine), so `git clone` on
   Kaggle gets it. If the repository is **private**, create a GitHub token that
   can read it. On each account: notebook editor → **Add-ons → Secrets → Add
   secret**, name `GITHUB_TOKEN`, paste the token, and enable it for this
   notebook. A public repository needs no token.

## Create the notebook (once per account)

1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** →
   upload `Matched_Experiments/kaggle/matched_experiment_kaggle.ipynb`.
2. Right-hand panel → **Session options**:
   - **Accelerator: GPU T4 x2.** Not P100: recent PyTorch builds no longer
     support it, and two T4s let two attacks run at once.
   - **Internet: On.**
3. **Add Input** → search **"NIH Chest X-rays"** (by *National Institutes of
   Health*, `nih-chest-xrays/data`, ~45 GB) → Add. Nothing is uploaded; the
   pipeline finds the images wherever Kaggle mounts them.
4. In the second cell set `CELL` (table above). Keep `SEEDS = "42"`.
5. Give the notebook a clear title, e.g. `matched-fgsm-noreg`.

Optional: set `SMOKE = True` and run it once (~25 min) to check that
everything works on Kaggle. Set it back to `False` afterwards. Smoke output
goes to `results_smoke/` and never mixes with real results.

## Run it

1. Click **Save Version** → **Save & Run All (Commit)** → Save.
   It runs in the background; you can close the browser.
2. The pipeline stops by itself after **11 hours**, so Kaggle's 12-hour limit
   never kills it and the output is always saved.
3. When the version finishes, open it → **Output** → `results/STATUS.md`:
   - **COMPLETE**: this account is done.
   - **NOT FINISHED**: continue as below.
   - **FAILED**: read `results/logs/<task>.log`, fix, and continue as below.

### Continue in the next session

1. Open the notebook editor → **Add Input** → **Your Work → Notebooks** →
   choose **this same notebook**. Its latest output is mounted as an input.
   On later sessions, make sure the attached output is the **latest version**:
   the input panel shows the version and offers an update.
2. **Save Version → Save & Run All** again.

The pipeline restores the attached output and continues: training resumes
from its last finished epoch, and attacks resume from their last finished
shadow model / classifier. Repeat until `STATUS.md` says **COMPLETE**.

## Rough time per account (seed 42, GPU T4 ×2 — estimates, not measured)

| Cell | Training | Attacks (both GPUs) | Likely sessions |
|---|---|---|---|
| A `std_reg`    | ~2–3 h | ~7 h  | 1–2 |
| B `fgsm_reg`   | ~5–7 h | ~7 h B + ~7 h B2, in parallel | 2–3 |
| C `std_noreg`  | ~4 h   | ~7 h  | 2 |
| D `fgsm_noreg` | ~8–10 h | ~7 h | 2 |

The free quota is about **30 GPU-hours per account per week**, so seed 42
fits in one week on every account. For more seeds, set `SEEDS = "42 43 44"`
in later weeks: finished seeds are skipped automatically. The reviewers asked
for repeated runs, so 3 seeds are worth it if time allows.

## Combine the four accounts (on your own machine)

1. On each account: open the final version → **Output** → **Download**
   (one zip). Or with the Kaggle CLI:
   `kaggle kernels output <username>/<notebook-slug> -p kaggle_out/<cell>`.
2. On your machine, from the repository root:

   ```bash
   python Matched_Experiments/kaggle/merge_kaggle_results.py \
       --out Matched_Experiments/results_kaggle \
       std_reg.zip fgsm_reg.zip std_noreg.zip fgsm_noreg.zip
   ```

3. Open `Matched_Experiments/results_kaggle/summary/REPORT.md`. It has the
   per-cell table, the paired contrasts (B−A, D−C, C−A, D−B, interaction,
   B2−B) and the gap-matched analysis. Model files are left out of the merge
   unless you pass `--with_models`.

**Important:** all four accounts must use the **same git commit**, so the
cells are comparable. Don't push code changes between the accounts' runs.
Each log records the commit in `results/logs/kaggle_pipeline.log` and in the
notebook's "Code version" line. Use T4 ×2 on all four accounts.

## What is where in a Kaggle output

```
results/
  STATUS.md, kaggle_state.json    progress of this account
  victims/<run>/                  model.pth, model_meta.json, history.csv, checkpoints/
  eval/<run>/                     evasion.json, threshold_mia.json
  attacks/{shadow_mia,afmia}/<run>/results.{json,txt}
  splits/attack_split_seed42.json exact attack pool / evaluation images
  logs/                           one log per task + kaggle_pipeline.log
  summary/REPORT.md               this account's cell on its own (contrasts need the merge)
```
