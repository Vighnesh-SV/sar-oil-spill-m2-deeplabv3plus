# M2 Improved DeepLabV3+: group-disjoint (leak-free) split, SOS Sentinel-1

Same model, recipe and seed as the official run; only the split changes (the split used for M1: `leakage_check/splits/sos_sentinel_group_s0.csv`, train = partitions train+val+calib, 3,358 tiles). Test: 835 tiles from sea areas never seen in training. 80 epochs, 72 min.

| Metric (%, with TTA) | M2 official split (leaky) | **M2 group-disjoint** | Difference | M1 group-disjoint |
|---|---|---|---|---|
| **Oil IoU** | 78.65 | **77.33** | -1.32 | 76.68 |
| **Oil F1** | 88.05 | **87.22** | -0.83 | 86.80 |
| Precision (oil) | 86.64 | **88.03** | +1.39 | 86.59 |
| Recall (oil) | 89.50 | **86.42** | -3.08 | 87.01 |
| mIoU | 83.20 | **83.62** | +0.41 | 83.06 |
| IoU (background) | 87.76 | **89.91** | +2.15 | 89.43 |
| Accuracy | 91.56 | **92.49** | +0.93 | 92.16 |
| Boundary F1 (2 px) | 41.12 | **47.53** | +6.41 | 45.52 |
| ECE (×100) | 3.16 | **2.41** | -0.75 | 2.64 |
| Oil share of test pixels | 34.72 | **29.64** | -5.08 | 29.64 |

## Interpretation
- **The leak inflates M2's oil scores slightly** (oil IoU -1.32 pp, F1 -0.83 pp): the same small effect as for M1. Recall falls, precision rises.
- mIoU rises because the leak-free test set has less oil; oil IoU and F1 are the fair comparison.
- On the leak-free split M2 is slightly ahead of M1 (oil IoU +0.65 pp, F1 +0.41 pp) with 5× fewer parameters. Single seed for both.

Figures: `training_curve.png`, `per_tile_iou.png` (median 0.69; 4 % of oil tiles below 0.3), `qualitative.png`.
