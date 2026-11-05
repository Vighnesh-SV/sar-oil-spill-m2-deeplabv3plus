# M1 CBD-Net: external test on Peru (new sea, never used for training)

**Data:** Peru set [N22], corrected version: 2,013 Sentinel-1 patches (512 × 512) from 40 scenes, 2014–2024. Labels: oil, look-alike, ship, land, sea. Scoring is binary (oil vs everything else) with the same pooled metrics and 8-way TTA as the SOS test. Each patch is cut into four 256 × 256 tiles, the training size.

## SOS (in-domain) vs Peru (new sea), all 2,013 patches

| Metric (%) | Official model: SOS | **Official model: Peru** | Leak-free model: SOS | **Leak-free model: Peru** |
|---|---|---|---|---|
| **Oil F1** | 87.9 | **24.5** | 86.8 | **29.4** |
| **Oil IoU** | 78.4 | **13.9** | 76.7 | **17.3** |
| Precision (oil) | 84.4 | **14.0** | 86.6 | **17.4** |
| Recall (oil) | 91.7 | **96.8** | 87.0 | **95.3** |
| mIoU | 82.8 | **37.2** | 83.1 | **43.5** |
| Accuracy | 91.2 | **62.8** | 92.2 | **71.6** |
| ROC-AUC (×100) | 97.5 | **92.1** | 97.5 | **92.6** |
| ECE (×100) | 2.8 | **26.5** | 2.6 | **18.6** |

## What the model calls oil on Peru (share of each true class predicted as oil, %)

| True class | Share of Peru pixels | Official model | Leak-free model |
|---|---|---|---|
| oil | 6.2 | 96.8 | 95.3 |
| look-alike | 5.3 | 89.9 | 86.0 |
| ship | 0.1 | 68.6 | 45.2 |
| sea | 84.1 | 37.6 | 27.6 |
| land | 4.3 | 14.3 | 7.7 |

Peru's own 338-patch test subset (5 scenes): oil F1 16.9 / 20.0, oil IoU 9.2 / 11.1 (official / leak-free model).

## Findings
1. **Severe drop under domain shift:** oil F1 falls from about 87 % on SOS to 24–29 % on Peru.
2. **Recall stays high (95–97 %) but precision collapses (14–17 %).** The model calls almost every dark area oil: 86–90 % of look-alike pixels, 45–69 % of ship pixels and 28–38 % of open sea. About 6 of every 7 oil alarms are false.
3. **The ranking is still good (ROC-AUC 0.92–0.93), but the 0.5 threshold and the probabilities are wrong**: ECE rises from about 0.03 to 0.19–0.26. The model is confidently wrong on the new sea. This is direct evidence for the calibration (C2) and alert-threshold (C3) contributions.
4. The leak-free model transfers slightly better (F1 +5 pp, ECE lower). This is a single seed.
5. Not a data bug: Peru and SOS have similar sea brightness (median 86 vs 89). Peru has lower contrast and darker oil, which is genuine domain shift.

## Qualitative (`qualitative_official.png`, `qualitative_groupsplit.png`)
- **Oil patches:** the slicks are found (green), but the model spreads oil into the surrounding dark sea and along the coastline (red).
- **Look-alike patches** (whole tiles of low-wind or other dark water) are predicted as oil almost everywhere. SOS has no look-alike class, so the model never learned to reject them.
