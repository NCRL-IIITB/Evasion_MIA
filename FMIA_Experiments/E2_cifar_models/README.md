# E2 — Datasets × victim architectures (CIFAR-10, CIFAR-100)

**Question.** Does CF-MIA beat the existing attacks for every victim architecture and on a harder dataset, for standard and adversarially trained victims?

## Design

| Dataset | Architectures | Recipes | Models per family |
|---|---|---|---|
| CIFAR-10 | ResNet-18 (`r18`), WRN-16-4 (`wrn`), VGG-11 (`vgg`) | standard, N-FGSM ε = 8/255 | 10 (2 sources) |
| CIFAR-100 | ResNet-18, WRN-16-4 | standard, N-FGSM ε = 8/255 | 10 (2 sources) |

That is 10 families and 100 models. Each trains on a random half of the 50k training images (50 epochs, augmentation).

Per family (3 targets each):
- `GB_<family>`: grey-box, same architecture and recipe references, transfer from the family's sources.
- `WB_<family>`: white-box, the victim's own gradients.
- `BB_<family>`: black-box, references of the other architectures of the same dataset and recipe, transfer from one of them.

## What to look at

- **§2 headline:** one row per (family, access level). The paper claim is "CF-MIA ≥ best existing in most rows".
- **CIFAR-100:** memorisation is much stronger, so LiRA is much stronger too. This is the hardest test for CF-MIA's margin.
- **Section 1:** the robust member − non-member gap per architecture. It explains where fluctuation helps.

## Cost

100 models, about 30–35 GPU-h in total, so **2–3 sessions**. Training order is round-robin over families, so a partial run still covers every family. Use `PARTIAL = True` for early numbers.

## Running on Kaggle (this account only)

1. **Account set-up:** phone-verify the account (GPU and internet need it).
2. **Code**, either way:
   - **Git:** the notebook clones `NCRL-IIITB/Evasion_MIA`. For a private repository, add a Kaggle secret `GITHUB_TOKEN` (Add-ons → Secrets).
   - **Dataset:** upload this folder (`FMIA_Experiments/E2_cifar_models`) as a private Kaggle Dataset and add it as an input. The notebook finds it.
3. **Create the notebook:** File → Import Notebook → `kaggle_notebook.ipynb` from this folder. Then set:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
4. **Smoke test:** set `SMOKE = True` → **Save Version → Save & Run All (Commit)**. Check `results_E2_cifar_models_smoke/STATUS.md` = COMPLETE (20–40 min).
5. **Full run:** set `SMOKE = False` → **Save & Run All (Commit)**. The run stops by itself after 11 h.
6. **Resume:** if `results_E2_cifar_models/STATUS.md` says *NOT FINISHED*: **Add Input → Your Work → Notebooks →** this notebook (latest version) → **Save Version** again. It restores the attached output and continues. Training resumes from the last epoch.
7. **Early look:** `PARTIAL = True` attacks with the models finished so far. Results are recomputed when more references arrive.

Lab GPU instead: `python run.py` in this folder (one worker per visible GPU).

## Output (`results_E2_cifar_models/`)

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
