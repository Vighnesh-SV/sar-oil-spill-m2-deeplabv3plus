# M2: Improved DeepLabV3+ (MobileNetV2 + scSE)

**Paper:** J. Zhang, P. Yang, X. Ren, "Detection of oil spill in SAR image using an improved DeepLabV3+," *Sensors*, 2024 [P01].
**Code:** no official code. We re-implemented the network from the paper: `common/models.py` (`dlv3p_mnv2`).
**Full report:** [`report/M2_DeepLabV3Plus_Results_Report.pdf`](report/M2_DeepLabV3Plus_Results_Report.pdf) (Word version next to it).

## Results at a glance (SOS Sentinel-1, official split, 839 test tiles, TTA)

| | **Ours (M2)** | Published [P01] | Gap | M1 CBD-Net (ours) |
|---|---|---|---|---|
| **mIoU** | **83.20 %** | 81.34 % | +1.86 pp | 82.79 % |
| **F1 (oil)** | **88.05 %** | 89.62 % | −1.57 pp | 87.91 % |
| Parameters | 6.04 M | 5.84 M | | 31.4 M |

**Verdict:** the re-implementation succeeds (both gaps are within ±2 pp).

## All tests

| Test | Official-split model | Leak-free model | M1 (official / leak-free) |
|---|---|---|---|
| SOS, oil F1 | 88.05 % | 87.22 % (leak-free test set) | 87.91 / 86.80 % |
| SOS, oil IoU | 78.65 % | 77.33 % | 78.43 / 76.68 % |
| Peru, oil F1 | 22.9 % | 22.3 % | 24.5 / 29.4 % |
| Peru, look-alike called oil | 87.4 % | 90.4 % | 89.9 / 86.0 % |
| Peru, ECE | 0.296 | 0.300 | 0.265 / 0.186 |
| MDOD, oil boxes flagged | 57.0 % | 50.5 % | 64.0 / 55.5 % |
| MDOD, AUC oil vs look-alikes | 0.293 | 0.286 | 0.282 / 0.335 |

**Findings:** M2 matches or slightly beats M1 in-domain with 5× fewer parameters, and the leak effect is small (about 1 pp). On new seas it fails exactly like M1: dark areas and look-alikes are called oil, and the probabilities are over-confident. The failure is shared by two different architectures.

## Setup
Adam, lr 1e-3, batch 15, 80 epochs, BCE + Dice (paper). Assumed where the paper is silent: cosine learning-rate decay, flip/rot90 augmentation, ImageNet weights, cSE reduction 16, 256 ASPP channels, input scaling [0, 255] → [−1.6, 1.6] (the CBD-Net scaling, used for all methods, instead of ImageNet mean/std). The final-epoch model is evaluated. Training: about 1 h 11 min per run on an RTX 3050 (4 GB). Training saves its full state every epoch (`last.pth`) and can continue with `--resume`.

## Files

| File | What it is |
|---|---|
| `make_report_m2.py` | Builds `results/` from the finished runs |
| `results/sentinel_official_split/` | Test A: results.md, figures, metrics |
| `results/sentinel_group_split/` | Test B (leak-free split) |
| `results/peru_external/`, `results/mdod_external/` | Tests C and D, both models |
| `report/` | Word + PDF report, its builder (`build_m2_report.py`, `to_pdf.ps1`) and terminal figures |
| `logs/` | Console logs of every run |
| `../runs/M2_dlv3p_mnv2_sentinel_s0[_groupsplit]/` | Weights (24 MB), probability maps, predicted masks |

## Reproduce (from `experiments/`)
```powershell
$S = "leakage_check\splits\sos_sentinel_group_s0.csv"
.venv\Scripts\python common\train.py --arch dlv3p_mnv2 --resume
.venv\Scripts\python common\evaluate.py --run runs\M2_dlv3p_mnv2_sentinel_s0 --arch dlv3p_mnv2
.venv\Scripts\python common\train.py --arch dlv3p_mnv2 --split $S --resume
.venv\Scripts\python common\evaluate.py --run runs\M2_dlv3p_mnv2_sentinel_s0_groupsplit --arch dlv3p_mnv2 --split $S
.venv\Scripts\python common\evaluate_peru.py --run runs\<run> --arch dlv3p_mnv2
.venv\Scripts\python common\evaluate_mdod.py --run runs\<run> --arch dlv3p_mnv2
.venv\Scripts\python M2_DeepLabV3Plus\make_report_m2.py
```

## Run from an editor (Antigravity / VS Code)

1. Open the repository folder. `.vscode/settings.json` selects the project Python (`.venv\Scripts\python.exe`).
2. Open `M2_DeepLabV3Plus/run_m2.py`, set `STEP` (`check`, `train`, `eval_sos`, `eval_peru`, `eval_mdod`, `report`, `all_tests`) and `SPLIT` (`leaky` or `leakfree`) at the top, and press Run.
3. `check` (the default) only lists what exists and prints the commands. Training refuses to replace an existing model unless `OVERWRITE = True`.
