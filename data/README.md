# Datasets (not included)

The datasets are **not** in this repository: they are large and their licences do not allow redistribution.
Download them from the original sources and place them here (or set the environment variables).

| Dataset | Role | Source | Licence | Put it at | Or set |
|---|---|---|---|---|---|
| **SOS** (Sentinel-1 + ALOS PALSAR, 256 × 256 tiles) | training + in-domain test | link in the CBD-Net README: github.com/CUG-URS/CBDNet-main | academic use only | `data/SOS_dataset/` (with `train/` and `test/`) | `SOS_ROOT` |
| **Peru** (2,013 Sentinel-1 patches, 512 × 512, 40 scenes) | test only (new sea, look-alikes) | IEEE DataPort, doi 10.21227/y7qb-we83, file `peru_event_region_v1_final.zip` | see DataPort page | `data/Peru/peru_event_region_v1_jan25_copy/` | `PERU_ROOT` |
| **MDOD** (4,673 Sentinel-1 scenes, 24,940 boxes) | test only (false alerts per look-alike type) | OSF project osf.io/vnmkj, `Dataset.zip.001` + `.002` (open `.001` with 7-Zip) | CC BY-NC | `data/MDOD/Dataset/` (with `image/` and `label/`) | `MDOD_ROOT` |

Expected layout:
```
data/
  SOS_dataset/train/sentinel/<id>_sat.jpg, <id>_mask.png
  SOS_dataset/test/sentinel/sat/<id>_sat.jpg, gt/<id>_mask.png
  Peru/peru_event_region_v1_jan25_copy/manifests/patch_manifest.tsv, ...
  MDOD/Dataset/image/<id>.png, label/label.txt
```

Label conventions used by the code:
- SOS: mask 255 = oil, 0 = background (stored with 3 identical channels; read as one channel).
- Peru: 0 sea, 1 oil, 2 look-alike, 3 ship, 4 land (1 and 2 confirmed by the dataset's correction log; 0/3/4 assumed). Scored as oil (1) vs everything else.
- MDOD `category_id`: 1 biogenic slick, 2 low wind, 3 oil spill, 4 rain cell, 5 sea ice (inferred from the MDOD paper's class order and example figures; the release has no class names).

The leak-free SOS split (`leakage_check/splits/sos_sentinel_group_s0.csv`) stores paths relative to `data/SOS_dataset/` (for example `test/sentinel/sat/20001_sat.jpg`).
