# FMIA experiments — can fluctuation-based MIA beat the existing attacks?

Five independent experiments.  **Each folder runs on its own Kaggle account** and
needs nothing from any other folder (the shared code is copied into each one).

| Folder | Question | Data / victims | Models | Rough cost (T4 ×2) |
|---|---|---|---|---|
| [E1_access_levels](E1_access_levels/) | White- vs grey- vs black-box; wrong / unknown training recipe; the paper's AT-shadow recipe | CIFAR-10, ResNet-18 std / AT + mixed-architecture references | 58 | 2 sessions |
| [E2_cifar_models](E2_cifar_models/) | Does it hold across architectures and a harder dataset? | CIFAR-10 (ResNet-18, WRN-16-4, VGG-11), CIFAR-100 (ResNet-18, WRN-16-4), std / AT | 100 | 2–3 sessions |
| [E3_medical_datasets](E3_medical_datasets/) | Does it hold on medical images (the paper's domain)? | ChestMNIST (= NIH ChestX-ray14), DermaMNIST, PathMNIST, BloodMNIST at 64 px | 128 | 2 sessions |
| [E4_fluctuation_variants](E4_fluctuation_variants/) | Which fluctuation implementation works, and why? | CIFAR-10, ResNet-18 std / AT, large probe bank | 40 | 2 sessions |
| [E5_at_strength](E5_at_strength/) | Robustness–privacy trade-off: AT ε ∈ {2, 4, 8, 16}, PGD, TRADES | CIFAR-10, ResNet-18 | 54 | 2–3 sessions |

Session = one ≤ 11 h Kaggle run; every run resumes from the previous output.  Costs
are estimates: the pipeline measures nothing in advance, it simply continues next session.

## Why the protocol changed from v1–v3

The earlier experiments compared **one** victim with 8 shadows against **one** other
victim.  Reviewers of MIA papers expect the protocol of LiRA (Carlini et al., S&P 2022)
and RMIA (Zarifzadeh et al., ICML 2024):

1. Train a **zoo** of models of one family, each on a random half of a population.
   Every record is then a member of about half of the models.
2. Each model in turn is the **target**; the others are the attacker's
   **reference (shadow) models**, whose membership the attacker knows.
3. Report **TPR at 1 % and 0.1 % FPR** and AUC, mean ± std over several targets.
   Accuracy at a 0.5 threshold, the old paper's metric, is not accepted any more.

Every model is probed **once**: clean, augmented, noisy and adversarial queries.  The
results are cached as per-record statistics, so the access levels, attack variants
and baselines are cheap numpy computations on the same models.

## The attacks

**Existing (baselines):**
- Loss threshold (Yeom 2018)
- Modified entropy (Song & Mittal 2021)
- Shadow-model MIA, the paper's "Original MIA" (Shokri 2017)
- LiRA offline / online (Carlini 2022)
- RMIA offline / online (Zarifzadeh 2024)
- Adversarial-loss threshold (Song et al. 2019): the robustness-based attack built for adversarially trained models; a fluctuation paper must beat this one
- Label-only augmentation/noise (Choquette-Choo 2021)
- Gradient norm (white-box)

**Fluctuation (ours):**
- `fmia_paper`: the submitted paper's attack. Gradient boosting on per-class |p_clean − p_adv| + max + argmax.
- `fl_raw`, `fl_cal_off`, `fl_cal_on`: one fluctuation statistic, raw or calibrated per example.
- `fl_gb`: gradient boosting on all fluctuation features.
- **`cfmia_on` / `cfmia_off`: CF-MIA, Calibrated multi-probe Fluctuation MIA.** This is the proposed attack:
  1. Probe each record several ways: transfer FGSM/PGD at several ε, Gaussian noise, or the victim's own gradient in white-box.
  2. For every probe, compute fluctuation statistics: the logit drop φ_clean − φ_probe, the paper's Σ|Δp|, and the variance under noise.
  3. **Calibrate each statistic per record** against the reference models. Offline: a z-score against the OUT models. Online: a Gaussian likelihood ratio IN vs OUT, the LiRA idea applied to fluctuation.
  4. **Combine them** with a logistic regression learned on reference models acting as pseudo-targets.

  The `_fl` variants use fluctuation statistics only. The full variants add the confidence statistic LiRA uses, so CF-MIA contains LiRA as a special case.

Everything an attack tunes (weights, probe, sign, RMIA's *a*) is fitted on reference
models only; the target's labels are used only for scoring.

### What would count as a publishable result

| Claim | Evidence | Where |
|---|---|---|
| CF-MIA ≥ strongest existing attack | `cfmia_on` vs the best of LiRA/RMIA online; `cfmia_off` vs the best offline baseline, in TPR@1% | §2 of every report |
| The gain comes from fluctuation, not from "more features" | `cfmia_fl_*` (fluctuation only) is competitive, and `lira_aug_on` (18 benign queries) does **not** close the gap | E4; every report |
| The gain grows with adversarial training | Δ(CF-MIA − LiRA) increases with ε while evasion success falls | E5 trade-off plot |
| It holds across settings and is practical | Gain on CIFAR-100 / other architectures / medical data; robust to wrong recipe and black-box | E1–E3 |
| Calibration explains why the paper's attack fails on standard models | `fmia_paper` → `fl_cal_on` → `cfmia_on` on `r18_std` | E4 |

If CF-MIA beats LiRA/RMIA only on adversarially trained victims, that is still a
clean paper: *fluctuation is the right signal for robust models*.

## Running

1. **Code:** commit and push this folder (the notebook clones the repository), or upload a single `E*` folder as a private Kaggle Dataset and attach it.
2. **Notebook:** in each account, import `E*/kaggle_notebook.ipynb` (File → Import Notebook). Then set:
   - Accelerator **GPU T4 ×2**
   - Internet **On**
3. **Smoke test:** run once with `SMOKE = True`.
4. **Full run:** set `SMOKE = False` and use **Save & Run All (Commit)**.
5. **Resume:** if `STATUS.md` says *NOT FINISHED*, attach the notebook's latest output as an input and commit again.
6. **Results:** `results_<folder>/summary/REPORT.md`, `leaderboard.csv`, `attack_results.csv`, figures.

Local / lab GPU: `cd FMIA_Experiments/E1_access_levels && python run.py` (writes `results_E1_access_levels/`).
CPU check without downloads: `FMIA_ARCH_OVERRIDE=simplecnn python run.py --smoke --dataset_override synthetic`.

## Editing the code

The copies in `E*/fmia/` and `E*/run.py` come from `_shared/`.  Edit `_shared/`, then run
`python FMIA_Experiments/_shared/sync.py` (`--check` reports stale copies).  Only
`E*/config.py` and `E*/README.md` are per-experiment.

```
_shared/fmia/
  data.py      CIFAR-10/100 (torchvision), MedMNIST v3 (pip medmnist), synthetic; GPU augmentation
  models.py    ResNet-18/34, WRN-16-4/28-2, VGG-11/16, MobileNetV2, DenseNet-121, small CNN
  adv.py       losses, φ (LiRA logit scaling), input gradients under AMP, FGSM/PGD, N-FGSM/FGSM-RS/TRADES
  train.py     resumable training (std, nfgsm, fgsm_rs, pgd, trades), last-epoch model
  signals.py   probe engine (one pass per model) + evasion / robust-overfitting diagnostics
  attacks.py   all attacks on cached signals; per-example calibration; CF-MIA
  runner.py    settings × targets, resumable; report.py  REPORT.md, CSVs, figures
  plan.py      config → models, keep masks, probes, settings
_shared/run.py  driver: restore output → train/probe on both GPUs → attacks → report
```
