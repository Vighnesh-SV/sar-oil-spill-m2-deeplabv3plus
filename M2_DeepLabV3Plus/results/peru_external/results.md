# M2 Improved DeepLabV3+: external test on Peru (new sea)

Same protocol as M1: all 2,013 Peru patches, oil vs everything else, 8-way TTA, frozen models.

| Metric (%) | M2 official: SOS | **M2 official: Peru** | M2 leak-free: SOS | **M2 leak-free: Peru** | M1 official: Peru | M1 leak-free: Peru |
|---|---|---|---|---|---|---|
| **Oil F1** | 88.0 | **22.9** | 87.2 | **22.3** | 24.5 | 29.4 |
| **Oil IoU** | 78.6 | **12.9** | 77.3 | **12.6** | 13.9 | 17.3 |
| Precision (oil) | 86.6 | **13.0** | 88.0 | **12.7** | 14.0 | 17.4 |
| Recall (oil) | 89.5 | **94.8** | 86.4 | **94.2** | 96.8 | 95.3 |
| mIoU | 83.2 | **35.3** | 83.6 | **34.6** | 37.2 | 43.5 |
| ROC-AUC (×100) | 97.2 | **90.8** | 97.7 | **90.0** | 92.1 | 92.6 |
| ECE (×100) | 3.2 | **29.5** | 2.4 | **30.0** | 26.5 | 18.6 |

| True class | Called oil: M2 official | Called oil: M2 leak-free | M1 official | M1 leak-free |
|---|---|---|---|---|
| oil | 94.8 | 94.2 | 96.8 | 95.3 |
| look-alike | 87.4 | 90.4 | 89.9 | 86.0 |
| ship | 61.0 | 63.9 | 68.6 | 45.2 |
| sea | 40.7 | 41.8 | 37.6 | 27.6 |
| land | 13.5 | 11.0 | 14.3 | 7.7 |

## Findings
1. **M2 transfers as badly as M1**: oil F1 22.9 / 22.3 % (M1: 24.5 / 29.4 %). Recall stays about 95 %, precision about 13 %.
2. **Look-alikes are called oil** (87.4–90.4 %), and the probabilities are over-confident (ECE about 0.30).
3. Unlike M1, the leak-free M2 is **not** better on Peru. The M1 gain (+5 pp F1) may therefore be seed noise rather than an effect of the split.
4. The failure is shared by two different architectures, so it comes from the training data (SOS has no look-alike class), not from one model design.
