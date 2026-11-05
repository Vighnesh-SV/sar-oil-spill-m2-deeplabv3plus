# M2 Improved DeepLabV3+: results on SOS Sentinel-1 (official split)

Run: `runs/M2_dlv3p_mnv2_sentinel_s0`. Test tiles: 839. Parameters: 6.04 M. Trained 80 epochs (70 min); the final-epoch model is evaluated (final loss 0.3098).

## Quantitative results (%, pixels pooled over all test tiles)

| Metric | Ours, with TTA | Ours, single pass | Published [P01] | M1 CBD-Net (ours, TTA) |
|---|---|---|---|---|
| Accuracy | 91.56 | 91.43 | — | 91.24 |
| Precision (oil) | 86.64 | 86.36 | — | 84.43 |
| Recall (oil) | 89.50 | 89.43 | — | 91.69 |
| **F1 (oil)** | 88.05 | 87.87 | 89.62 | 87.91 |
| IoU (oil) | 78.65 | 78.36 | — | 78.43 |
| IoU (background) | 87.76 | 87.57 | — | 87.15 |
| **mIoU** | 83.20 | 82.97 | 81.34 | 82.79 |
| Boundary F1 (2 px) | 41.12 | 40.64 | — | 37.82 |
| F1 as computed by the official script (background class) | 93.48 | 93.37 | — | 93.14 |

| Probability quality | With TTA | Single pass | M1 (TTA) |
|---|---|---|---|
| ROC-AUC | 0.9720 | 0.9711 | 0.9748 |
| PR-AUC | 0.9557 | 0.9546 | 0.9585 |
| ECE (15 bins) | 0.0316 | 0.0357 | 0.0283 |
| Brier score | 0.0632 | 0.0647 | 0.0639 |

**Reproduction gap:** mIoU +1.86 pp, F1 -1.57 pp (target: within ±2 pp). The paper does not say which class its F1 is computed for; if it used the official SOS script (background class), the comparable value is 93.48 %.

## Figures
- `training_curve.png`: training loss per epoch.
- `per_tile_iou.png`: per-tile oil IoU. Median 0.70; 4 % of oil tiles score below 0.3.
- `qualitative.png`: four median and four worst test tiles (green = correct oil, red = false oil, blue = missed oil).
