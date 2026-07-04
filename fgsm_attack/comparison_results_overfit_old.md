# Model Comparison: Clean vs Adversarial

| Metric | Clean | FGSM (0.001) | FGSM (0.002) | FGSM (0.005) | FGSM (0.01) |
| --- | --- | --- | --- | --- | --- |
| Hamming Loss | 0.0775 | 0.1751 | 0.1945 | 0.2193 | 0.2348 |
| Exact Match | 0.4659 | 0.0005 | 0.0000 | 0.0000 | 0.0000 |
| F1 Macro | 0.5411 | 0.2686 | 0.2183 | 0.1535 | 0.1151 |
| Flip Rate | — | 0.0993 | 0.1190 | 0.1444 | 0.1604 |