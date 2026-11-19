# E4 — Which implementation of fluctuation-based MIA works, and why

**Question.** Which design choices make a fluctuation attack strong? This experiment gives the evidence that the gain comes from fluctuation and not from extra features or extra queries.

## Design

CIFAR-10, ResNet-18 standard (`r18_std`) and N-FGSM ε = 8/255 (`r18_at`). There are 20 models per family: 2 sources plus 18 targets/references. Every model is probed with 36 probes:

| Probe family | Variants |
|---|---|
| Transfer FGSM (true label) | ε ∈ {0.5, 1, 2, 4, 8, 16}/255 from std sources; {1, 2, 4, 8} from AT sources |
| Transfer FGSM, **own label** (the paper's label-free probe) | ε ∈ {2, 8}, std and AT sources |
| Transfer PGD-5 | ε ∈ {4, 8}, std and AT sources |
| Gaussian / uniform noise (8 draws) | σ ∈ {0.005, 0.01, 0.02, 0.04}; uniform ε ∈ {4, 8} |
| White-box FGSM / PGD-10 / own-label FGSM | ε up to 16/255 |
| Benign | flip, 18 LiRA augmentations |

Settings, for each victim type `t` (4 targets; 3 for the reference-count sweep):

| Setting | Probes allowed |
|---|---|
| `GB_t_all` | everything except white-box |
| `GB_t_paperprobe` | **only the paper's probe**: what calibration + stacking add to the paper's attack |
| `GB_t_fgsm` / `GB_t_pgd` / `GB_t_ownlabel` / `GB_t_noise` | one probe family |
| `WB_t_all` / `WB_t_self` | white-box |
| `GB_t_refs@{2,4,8,17}refs` | all probes, varying the number of reference models |

## What to look at

- **The ladder:** `fmia_paper` → `fl_raw` → `fl_cal_off` / `fl_cal_on` → `cfmia_fl_*` → `cfmia_*`. In `GB_t_paperprobe` it isolates calibration and stacking on the same probe.
- **"Single fluctuation statistics" table** of each setting: which probe, ε and statistic carries the signal (logit drop vs Σ|Δp| vs noise variance).
- **More queries alone don't explain the gain:** compare `lira_aug_on` (18 benign queries) with `cfmia_on`.
- **§5:** TPR@1% vs number of references for LiRA, RMIA and CF-MIA. This matters for the low-cost regime that RMIA targets.
  - CF-MIA learns its weights on references acting as pseudo-targets, so it needs **≥ 4 references (online) / ≥ 3 (offline)**. The 2-reference column shows only LiRA/RMIA.
  - Its cost in the low-reference regime is part of the honest comparison.

## Cost

40 models, about 13 GPU-h of training, plus heavy probing of 36 models (about 9 GPU-h). Expect **2 sessions**.

## Running on Kaggle (this account only)

1. **Account set-up:** phone-verify the account (GPU and internet need it).
2. **Code**, either way:
   - **Git:** the notebook clones `NCRL-IIITB/Evasion_MIA`. For a private repository, add a Kaggle secret `GITHUB_TOKEN` (Add-ons → Secrets).
   - **Dataset:** upload this folder (`FMIA_Experiments/E4_fluctuation_variants`) as a private Kaggle Dataset and add it as an input. The notebook finds it.
3. **Create the notebook:** File → Import Notebook → `kaggle_notebook.ipynb` from this folder. Then set:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
4. **Smoke test:** set `SMOKE = True` → **Save Version → Save & Run All (Commit)**. Check `results_E4_fluctuation_variants_smoke/STATUS.md` = COMPLETE (20–40 min).
5. **Full run:** set `SMOKE = False` → **Save & Run All (Commit)**. The run stops by itself after 11 h.
6. **Resume:** if `results_E4_fluctuation_variants/STATUS.md` says *NOT FINISHED*: **Add Input → Your Work → Notebooks →** this notebook (latest version) → **Save Version** again. It restores the attached output and continues. Training resumes from the last epoch.
7. **Early look:** `PARTIAL = True` attacks with the models finished so far. Results are recomputed when more references arrive.

Lab GPU instead: `python run.py` in this folder (one worker per visible GPU).

## Output (`results_E4_fluctuation_variants/`)

| Path | Content |
|---|---|
| `summary/REPORT.md` | models, evasion, **headline CF-MIA vs strongest existing attack**, every attack per setting, ROC figures |
| `summary/leaderboard.csv` | mean ± std per (setting, attack) |
| `summary/attack_results.csv` | one row per (setting, target, attack), including single-statistic rows |
| `attacks/<setting>__<target>.json` | raw metrics, chosen probes, RMIA *a*, references used |
| `models/<id>/` | weights (fp16), `meta.json` (recipe, history, train/test metric), `keep.npy` (membership) |
| `signals/<id>.npz` | cached per-record statistics (re-run attacks without a GPU: `python run.py --partial`) |
| `evasion/<id>.json` | clean / FGSM / PGD test metric, member vs non-member clean and robust correctness |
| `STATUS.md`, `logs/` | progress, one log per task |

Do not change `config.py` between sessions of one run. Model seeds and membership masks derive from the model names, but probes and settings must stay identical across sessions.
