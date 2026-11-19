# E1 — White-box vs grey-box vs black-box

**Question.** How much does each kind of attacker knowledge help fluctuation-based MIA, compared with the existing attacks, on a standard and on an adversarially trained victim?

The submitted paper claims a strict black-box attacker that does *not* know the training procedure. This experiment tests exactly that.

## Design

CIFAR-10. Each model trains on a random half of the 50k training images, for 50 epochs with crop + flip augmentation and no early stopping.

| Family | Models | Role |
|---|---|---|
| `r18_std` | 18 (2 sources) | ResNet-18, standard training: victims + grey/white-box references |
| `r18_at` | 18 (2 sources) | ResNet-18, N-FGSM adversarial training ε = 8/255: victims + references |
| `mix_std` | 11 (2 sources) | VGG-11 / WRN-16-4 / MobileNetV2, standard: black-box references |
| `mix_at` | 11 (2 sources) | same architectures, N-FGSM ε = 8/255: black-box references |

*Sources* are surrogate models whose gradients build the transfer probes. They are never targets or references.

Settings, for each victim type `t ∈ {std, at}`, with 4 targets each:

| Setting | Attacker knows | References | Probes |
|---|---|---|---|
| `WB_t` | weights | same recipe | victim's own FGSM/PGD + transfer + noise |
| `GB_t` | architecture + recipe | same recipe | transfer from same-recipe ResNet-18s + noise |
| `GB_t_recipeX` | architecture, **wrong recipe** | other recipe | transfer from other-recipe ResNet-18s |
| `GB_t_recipeMix` | architecture, recipe unknown | both recipes | both |
| `BB_t` | nothing | other architectures, both recipes | transfer from other architectures |
| `BB_t_queryonly` | nothing, no surrogates | other architectures | noise / augmentation only |
| `BB_t_paper` | the paper's recipe | AT shadows of other architectures | transfer from AT shadows |

## What to look at

- **§2 headline:** Δ TPR@1% of CF-MIA over the best existing attack, per access level.
- **The paper's claims, `BB_at_paper` vs `BB_std_paper`:**
  - Does fluctuation leak more on the AT victim?
  - Does `fmia_paper` beat `shokri` (the "Original MIA")?
  - Does CF-MIA improve both?
- **Recipe robustness:** `GB_t_recipeX` vs `GB_t`.
- **Cost of knowledge:** `WB_t` vs `GB_t` vs `BB_t`. White-box should be the ceiling.

## Cost

58 models (about 15–20 GPU-h of training) plus probing 50 models (about 5–7 GPU-h), using both T4s. Expect **2 sessions**. The attacks run on CPU at the end (about 1–2 h).

## Running on Kaggle (this account only)

1. **Account set-up:** phone-verify the account (GPU and internet need it).
2. **Code**, either way:
   - **Git:** the notebook clones `NCRL-IIITB/Evasion_MIA`. For a private repository, add a Kaggle secret `GITHUB_TOKEN` (Add-ons → Secrets).
   - **Dataset:** upload this folder (`FMIA_Experiments/E1_access_levels`) as a private Kaggle Dataset and add it as an input. The notebook finds it.
3. **Create the notebook:** File → Import Notebook → `kaggle_notebook.ipynb` from this folder. Then set:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
4. **Smoke test:** set `SMOKE = True` → **Save Version → Save & Run All (Commit)**. Check `results_E1_access_levels_smoke/STATUS.md` = COMPLETE (20–40 min).
5. **Full run:** set `SMOKE = False` → **Save & Run All (Commit)**. The run stops by itself after 11 h.
6. **Resume:** if `results_E1_access_levels/STATUS.md` says *NOT FINISHED*: **Add Input → Your Work → Notebooks →** this notebook (latest version) → **Save Version** again. It restores the attached output and continues. Training resumes from the last epoch.
7. **Early look:** `PARTIAL = True` attacks with the models finished so far. Results are recomputed when more references arrive.

Lab GPU instead: `python run.py` in this folder (one worker per visible GPU).

## Output (`results_E1_access_levels/`)

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
