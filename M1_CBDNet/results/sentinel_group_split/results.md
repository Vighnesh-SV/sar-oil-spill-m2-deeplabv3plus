# M1 CBD-Net: group-disjoint split (C1 leakage test), SOS Sentinel

**Question:** how much does the official SOS split flatter the scores? Every official test tile has augmented copies of the same sea area in training (`leakage_check/REPORT.md`).

**Setup:** identical model, loss, schedule and seed to the official run. Only the split changes.
- Train: 3358 tiles (partitions train + val + calib), about the official 3,354, so training size is not a factor.
- Test: 835 tiles from sea areas never seen in training (0 shared tiles, 0 shared groups).
- Training stopped at epoch 50; best training loss 0.3670 at epoch 46.

## Results (%, with TTA)

| Metric | Official split (leaky) | **Group-disjoint** | Difference |
|---|---|---|---|
| **Oil IoU** | 78.43 | **76.68** | -1.74 |
| **Oil F1** | 87.91 | **86.80** | -1.11 |
| Precision (oil) | 84.43 | **86.59** | +2.16 |
| Recall (oil) | 91.69 | **87.01** | -4.67 |
| mIoU | 82.79 | **83.06** | +0.27 |
| IoU (background) | 87.15 | **89.43** | +2.28 |
| Accuracy | 91.24 | **92.16** | +0.91 |
| Boundary F1 (2 px) | 37.82 | **45.52** | +7.71 |
| ECE (×100) | 2.83 | **2.64** | -0.19 |
| Oil share of test pixels | 34.72 | **29.64** | -5.08 |

## Interpretation
- **The leakage effect is small:** oil IoU -1.74 pp and oil F1 -1.11 pp. On unseen areas the model finds slightly less oil (recall -4.67 pp).
- **mIoU hides the drop.** The group test set has less oil (29.64 % vs 34.72 % of pixels), so background IoU rises and offsets the loss in oil IoU. Oil IoU and F1 are the fair comparison here.
- Calibration (ECE) is unchanged.

## Caveats
1. **One seed.** Differences of 1–2 pp are close to run-to-run variation. Repeat with 3 seeds before claiming a number.
2. **Groups are sea areas (crops), not the 21 original scenes.** Scene-level leakage remains possible, so this is a lower bound. The Peru dataset's scene split can test it.
3. The two test sets differ (different tiles), so part of any difference comes from test composition.

## Figures
- `training_curve.png`, `per_tile_iou.png` (median 0.69; 6% of oil tiles below 0.3), `qualitative.png`: same format as the official run.
