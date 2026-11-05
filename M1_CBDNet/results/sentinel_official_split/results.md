# M1 CBD-Net: results on SOS Sentinel (official split)

Run: `runs/M1_cbdnet_sentinel_dicebce_s0`. Test tiles: 839. Parameters: 31.4 M. Training stopped at epoch 71, best training loss 0.3515 at epoch 66.

## Quantitative results (%, pixels pooled over all test tiles)

| Metric | Ours, with TTA | Ours, single pass | Published |
|---|---|---|---|
| Accuracy | 91.24 | 91.08 | — |
| Precision (oil) | 84.43 | 84.14 | — |
| Recall (oil) | 91.69 | 91.57 | — |
| **F1 (oil)** | 87.91 | 87.70 | 87.81 |
| IoU (oil) | 78.43 | 78.09 | — |
| IoU (background) | 87.15 | 86.92 | — |
| **mIoU** | 82.79 | 82.51 | 82.94 |
| Boundary F1 (2 px) | 37.82 | 37.24 | — |
| F1 as computed by the official script (background class) | 93.14 | 93.00 | — |

| Probability quality | With TTA | Single pass |
|---|---|---|
| ROC-AUC | 0.9748 | 0.9737 |
| PR-AUC | 0.9585 | 0.9569 |
| ECE (15 bins) | 0.0283 | 0.0329 |
| Brier score | 0.0639 | 0.0657 |

Published values: independent re-run of CBD-Net in [P29]. TTA = the official 8-way rotation/flip test-time augmentation.
**Reproduction gap:** mIoU -0.15 pp, F1 +0.10 pp (target: within ±2 pp).

## Figures

- `training_curve.png`: training loss per epoch; dashed lines mark learning-rate drops.
- `per_tile_iou.png`: per-tile oil IoU. Median 0.69; 3% of oil tiles score below 0.3.
- `qualitative.png`: four median and four worst test tiles (green = correct oil, red = false oil, blue = missed oil).
