## FGSM Evasion Attack: Adversarial (FGSM-trained defence)

Test set: 33412 non-member images (manifest.csv, 70/30 patient-level split)

| Metric | Clean | ε=0.001 | ε=0.002 | ε=0.005 | ε=0.01 | ε=0.02 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hamming_loss | 0.0777 | 0.1195 | 0.1276 | 0.0680 | 0.0420 | 0.0625 |
| exact_match | 0.4500 | 0.2083 | 0.1650 | 0.4946 | 0.6440 | 0.5178 |
| f1_macro | 0.2525 | 0.1399 | 0.1270 | 0.3285 | 0.5262 | 0.3965 |
| auc_macro | 0.7516 | 0.5115 | 0.4669 | 0.7935 | 0.9044 | 0.8264 |
| flip_rate | — | 0.0422 | 0.0504 | 0.0188 | 0.0540 | 0.0387 |
