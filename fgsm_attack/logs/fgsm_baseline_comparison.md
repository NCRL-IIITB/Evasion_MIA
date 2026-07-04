## FGSM Evasion Attack: Baseline (best-practice standard)

Test set: 33412 non-member images (manifest.csv, 70/30 patient-level split)

| Metric | Clean | ε=0.001 | ε=0.002 | ε=0.005 | ε=0.01 | ε=0.02 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hamming_loss | 0.1323 | 0.1999 | 0.2499 | 0.3194 | 0.3499 | 0.3624 |
| exact_match | 0.2795 | 0.1152 | 0.0387 | 0.0013 | 0.0000 | 0.0000 |
| f1_macro | 0.3318 | 0.2091 | 0.1490 | 0.0811 | 0.0578 | 0.0458 |
| auc_macro | 0.8278 | 0.6832 | 0.5842 | 0.4396 | 0.3591 | 0.3110 |
| flip_rate | — | 0.0744 | 0.1287 | 0.2052 | 0.2419 | 0.2612 |