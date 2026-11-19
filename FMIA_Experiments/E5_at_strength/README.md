# E5 — Robustness–privacy trade-off: adversarial-training strength and method

**Question.** This is the paper's central claim as a dose–response curve. As adversarial training gets stronger and evasion gets harder:
- Does membership leakage through fluctuation grow?
- Does confidence-based leakage shrink?
- Does CF-MIA stay the strongest attack along the whole curve?

## Design

CIFAR-10, ResNet-18, 50 epochs, last-epoch models:

| Family | Training | Models |
|---|---|---|
| `std` | standard (2 of them are the transfer sources) | 10 |
| `nfgsm2`, `nfgsm4`, `nfgsm8`, `nfgsm16` | N-FGSM at ε = 2, 4, 8, 16 /255 | 8 each |
| `pgd8` | PGD-5 adversarial training, ε = 8/255 | 6 (trained last) |
| `trades8` | TRADES β = 6, 5 steps, ε = 8/255 | 6 (trained last) |

Transfer probes always come from **standard** surrogates, so the attacker never needs to know the defence.

Per family (3 targets each):
- `GB_F`: references trained like the victim.
- `GB_F_recipeX`: standard references only, defence unknown.
- `WB_F`: white-box.

## What to look at

- **§4 trade-off figure.** x = test accuracy under PGD (robustness); y = TPR@1% of Shokri, LiRA, RMIA, Song'19, the paper's attack and CF-MIA. The paper's story needs the CF-MIA − LiRA gap to grow to the right.
- **Section 1:** member − non-member robust gap vs ε. This is the mechanism (robust overfitting).
- **Evasion table:** FGSM/PGD success per family. This is the evasion half of the trade-off.
- **TRADES:** it explicitly trains output stability on the training set, so it is the best case for fluctuation.

## Cost

54 models. PGD/TRADES cost about 3× N-FGSM per model. Expect about 2–3 sessions; the N-FGSM curve is ready after the first.

## Running on Kaggle (this account only)

1. **Account set-up:** phone-verify the account (GPU and internet need it).
2. **Code**, either way:
   - **Git:** the notebook clones `NCRL-IIITB/Evasion_MIA`. For a private repository, add a Kaggle secret `GITHUB_TOKEN` (Add-ons → Secrets).
   - **Dataset:** upload this folder (`FMIA_Experiments/E5_at_strength`) as a private Kaggle Dataset and add it as an input. The notebook finds it.
3. **Create the notebook:** File → Import Notebook → `kaggle_notebook.ipynb` from this folder. Then set:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
4. **Smoke test:** set `SMOKE = True` → **Save Version → Save & Run All (Commit)**. Check `results_E5_at_strength_smoke/STATUS.md` = COMPLETE (20–40 min).
5. **Full run:** set `SMOKE = False` → **Save & Run All (Commit)**. The run stops by itself after 11 h.
6. **Resume:** if `results_E5_at_strength/STATUS.md` says *NOT FINISHED*: **Add Input → Your Work → Notebooks →** this notebook (latest version) → **Save Version** again. It restores the attached output and continues. Training resumes from the last epoch.
7. **Early look:** `PARTIAL = True` attacks with the models finished so far. Results are recomputed when more references arrive.

Lab GPU instead: `python run.py` in this folder (one worker per visible GPU).

## Output (`results_E5_at_strength/`)

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
