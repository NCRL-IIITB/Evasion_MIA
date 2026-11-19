# E3 — Medical imaging datasets (MedMNIST v3, 64×64)

**Question.** Does CF-MIA beat the existing attacks on medical images, the paper's domain?

This includes **ChestMNIST**, which is the NIH ChestX-ray14 dataset of the submitted paper (112,120 chest X-rays, 14 labels, multi-label), at 64×64.

## Design

| Dataset | Task | Population (members ≈ half) | Test |
|---|---|---|---|
| `chestmnist@64` | NIH chest X-ray, 14 labels, multi-label | 30,000 of train+val | 10,000 |
| `dermamnist@64` | HAM10000 dermatoscopy, 7 classes | all ≈ 8,000 | 2,005 |
| `pathmnist@64` | colon histology, 9 classes | 30,000 of train+val | 7,180 |
| `bloodmnist@64` | blood-cell microscopy, 8 classes | all ≈ 13,600 | 3,421 |

Per dataset:
- ResNet-18 standard / N-FGSM ε = 4/255: 10 models each, the victims.
- VGG-11 standard / AT: 6 models each, black-box references.

Training: 40 epochs, SGD 0.05, crop + flip. For multi-label data, φ is the mean signed logit of the correct labels, and accuracy is replaced by mean per-label AUC.

Settings per dataset and recipe (3 targets each): `GB`, `WB`, `GB_recipeX` (wrong recipe), `BB` (VGG references of both recipes).

An optional DenseNet-121 (ImageNet-pretrained) family for ChestMNIST, the paper's exact architecture, is in `config.py`, commented out. It costs about 4× ResNet-18.

## What to look at

- **ChestMNIST rows:** the direct successor of the paper's NIH experiment, with the proper protocol.
- **DermaMNIST:** small data means strong memorisation and the largest leakage.
- **The member − non-member robust gap** (section 1) per dataset.

## Cost

128 small-image models, about 20 GPU-h, so about **2 sessions**. Data comes from Zenodo via `pip install medmnist`, which the notebook runs. If the download fails, attach the `*_64.npz` files as a Kaggle dataset; the loader searches `/kaggle/input`.

## Running on Kaggle (this account only)

1. **Account set-up:** phone-verify the account (GPU and internet need it).
2. **Code**, either way:
   - **Git:** the notebook clones `NCRL-IIITB/Evasion_MIA`. For a private repository, add a Kaggle secret `GITHUB_TOKEN` (Add-ons → Secrets).
   - **Dataset:** upload this folder (`FMIA_Experiments/E3_medical_datasets`) as a private Kaggle Dataset and add it as an input. The notebook finds it.
3. **Create the notebook:** File → Import Notebook → `kaggle_notebook.ipynb` from this folder. Then set:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
   - (MedMNIST is downloaded from Zenodo; internet must be on)
4. **Smoke test:** set `SMOKE = True` → **Save Version → Save & Run All (Commit)**. Check `results_E3_medical_datasets_smoke/STATUS.md` = COMPLETE (20–40 min).
5. **Full run:** set `SMOKE = False` → **Save & Run All (Commit)**. The run stops by itself after 11 h.
6. **Resume:** if `results_E3_medical_datasets/STATUS.md` says *NOT FINISHED*: **Add Input → Your Work → Notebooks →** this notebook (latest version) → **Save Version** again. It restores the attached output and continues. Training resumes from the last epoch.
7. **Early look:** `PARTIAL = True` attacks with the models finished so far. Results are recomputed when more references arrive.

Lab GPU instead: `python run.py` in this folder (one worker per visible GPU).

## Output (`results_E3_medical_datasets/`)

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
