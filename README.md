# M2: Improved DeepLabV3+ for oil-spill segmentation in Sentinel-1 SAR

Re-implementation of the **Improved DeepLabV3+** (J. Zhang, P. Yang, X. Ren, "Detection of oil spill in SAR image using an improved DeepLabV3+," *Sensors*, doi 10.3390/s24175460). **No official code was released**, so the network was rebuilt from the paper: MobileNetV2 encoder (output stride 16), scSE attention in every inverted-residual block and on the ASPP output, DeepLabV3+ decoder. 6.04 M parameters / 6.56 GMACs (paper: 5.84 M / 6.16).

Evaluated exactly like M1: official (leaky) and **leak-free** SOS splits, **cross-sea tests** on Peru and MDOD, pooled metrics with bootstrap confidence intervals. This repository also holds the **M1-vs-M2 comparison** and the combined documents (M1 reference results are included read-only; M1's code is in [sar-oil-spill-m1-cbdnet](https://github.com/Vighnesh-SV/sar-oil-spill-m1-cbdnet)).

## Results (TTA, oil class)

| Test | Trained on | Accuracy | Precision | Recall | **F1** | **IoU** | mIoU | ECE | MSE | R² |
|---|---|---|---|---|---|---|---|---|---|---|
| SOS official test (839) | official split (leaky) | 91.56 | 86.64 | 89.50 | **88.05** | **78.65** | 83.20 | 0.032 | 0.063 | 0.721 |
| SOS leak-free test (835) | leak-free split | 92.49 | 88.03 | 86.42 | **87.22** | **77.33** | 83.62 | 0.024 | 0.055 | 0.734 |
| Peru, all 2,013 patches | official split | 60.21 | 13.01 | 94.84 | 22.89 | 12.92 | 35.32 | 0.296 | 0.321 | −4.500 |
| Peru, all 2,013 patches | leak-free split | 59.17 | 12.66 | 94.23 | 22.32 | 12.56 | 34.59 | 0.300 | 0.330 | −4.644 |
| Published | official split | — | — | — | 89.62 | — | 81.34 | — | — | — |

MDOD (400 boxes per class): oil boxes flagged 57.0 % / 50.5 %, look-alikes 86–99 %; AUC oil vs look-alikes 0.293 / 0.286.

**Reading:** within ±2 pp of the published numbers (mIoU +1.86, F1 −1.57). The paper does not say which class its F1 is for, and its F1 and mIoU are hard to reconcile if F1 is the oil class (see docs). On the leak-free split M2 is slightly but significantly better than M1 on SOS (F1 +0.41 pp, 95 % CI [+0.07, +0.76]); on Peru it is clearly worse (F1 −7.1 pp). Same domain-shift failure as M1. Mild overfitting (train − test oil IoU gap +1.8 / +3.7 pp).

All tables: [`docs/`](docs/); full report: [`M2_DeepLabV3Plus/report/`](M2_DeepLabV3Plus/report/).

## Repository structure

```
M2_DeepLabV3Plus/     one-file runner, report builders, results/, logs/, report/
common/               data loading, model registry (M2 = dlv3p_mnv2 in models.py), generic trainer (train.py), evaluation
M1_CBDNet/            M1 reference only: report helpers and M1 results used for the comparison (read-only)
tools/                build_all_results.py (rebuilds docs/ from saved results), analysis_overfit.py (needs the weights)
third_party/CBDNet-main   official CBD-Net code (submodule; M2 uses its BCE + Dice loss)
leakage_check/        proof of the SOS leak, scripts, leak-free split CSVs
runs/<run>/           saved metrics, CSVs, logs, figures for M1 and M2 (no weights)
results/              summary.csv, analysis/analysis.json (overfitting, CIs, dataset statistics)
docs/                 all-results tables (M1, M2), analysis, dataset checks and critique
data/                 put the datasets here (see data/README.md; not included)
```

## Setup (Windows, NVIDIA GPU)

```powershell
git clone --recurse-submodules https://github.com/Vighnesh-SV/sar-oil-spill-m2-deeplabv3plus
cd sar-oil-spill-m2-deeplabv3plus
python -m venv .venv          # Python 3.12
.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```
If you cloned without `--recurse-submodules`, run `git submodule update --init` (this fetches the official CBD-Net code into `third_party/CBDNet-main`).

Then put the datasets in `data/` as described in [`data/README.md`](data/README.md). Pretrained ImageNet weights download automatically into `.cache/` on first use.

**Trained weights are not included** (about 100–125 MB each, and derived from academic-use-only data). Either train them with the runner below, or copy existing `best.pth` files into `runs/<run name>/`.

## Run

**From an editor:** open the repository folder, open `M2_DeepLabV3Plus/run_m2.py`, set `STEP` (`check`, `train`, `eval_sos`, `eval_peru`, `eval_mdod`, `all_tests`, `report`) and `SPLIT` (`leaky` / `leakfree`), press Run. Training takes about 1 h 10 min on an RTX 3050 (4 GB) and resumes from `last.pth` if interrupted.

**From a terminal** (repository root):
```powershell
.venv\Scripts\python common\train.py --arch dlv3p_mnv2 --resume
.venv\Scripts\python common\evaluate.py --run runs\M2_dlv3p_mnv2_sentinel_s0 --arch dlv3p_mnv2
.venv\Scripts\python common\train.py --arch dlv3p_mnv2 --split leakage_check\splits\sos_sentinel_group_s0.csv --resume
.venv\Scripts\python common\evaluate.py --run runs\M2_dlv3p_mnv2_sentinel_s0_groupsplit --arch dlv3p_mnv2 --split leakage_check\splits\sos_sentinel_group_s0.csv
.venv\Scripts\python common\evaluate_peru.py --run runs\<run> --arch dlv3p_mnv2
.venv\Scripts\python common\evaluate_mdod.py --run runs\<run> --arch dlv3p_mnv2
.venv\Scripts\python tools\build_all_results.py        # rebuild docs/ from the saved results
```

**Assumed where the paper is silent:** cSE reduction 16, 256 ASPP channels, 48-channel low-level projection, cosine learning-rate decay, flip / 90° rotation augmentation, ImageNet weights, input scaling [0, 255] → [−1.6, 1.6] (the CBD-Net scaling, used for all methods). From the paper: Adam, lr 1e-3, batch 15, 80 epochs, BCE + Dice; the final-epoch model is evaluated (the paper uses no validation set).

## Integrity checks and known limitations

Checked (details in `docs/Datasets_Checks_Critique_QA.docx`):
- **The official SOS split leaks.** Test tiles 20001–20839 are augmented copies of training areas (5 copies of 444 areas); 100 % of test tiles have copies in training. Proof and scripts: [`leakage_check/REPORT.md`](leakage_check/REPORT.md). Every model is therefore trained and tested twice: on the official (leaky) split, for comparison with the paper, and on a **group-disjoint (leak-free) split** that shares no sea area between partitions (verified: 0 shared areas in all partition pairs).
- Every SOS image has its mask (3,354/3,354 train, 839/839 test); no ID is in both train and test.
- SOS and Peru are greyscale stored as 3 identical channels and have similar brightness (mean 96 vs 88); MDOD is brighter (mean 135), about 10× coarser and has only boxes, so it is a **supplementary** look-alike stress test.
- The official evaluation script `t1p3-Iou.py` reports precision/recall/F1 for the **background** class (`labels=[255, 0]`). We report oil-class metrics as the main numbers and the script value separately.
- Test-time augmentation (8 flips/rotations) is inverted correctly; the decision threshold is fixed at 0.5; no test data is used for model selection.

Limitations:
- One training seed per model; SOS differences under about 1 pp are within noise (bootstrap CIs in the docs).
- The leak-free models were trained on train + val + calib (3,358 tiles), so no SOS data is held out for calibration. Any calibration or alert-threshold method must be fitted on Peru train (1,594 patches, 33 scenes) and reported only on Peru test (338 patches, 5 scenes).
- Peru classes 0/3/4 and the MDOD class IDs are inferred, not documented by the dataset authors (they affect only the per-class tables).

## Licence

Our code: MIT (see `LICENSE`). The official CBD-Net code is linked, not copied. The datasets are not included and keep their own licences.
