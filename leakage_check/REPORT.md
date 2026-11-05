# SOS train/test leakage check (3 Nov 2026)

Scripts: `near_duplicates.py`, `assign_block2.py`, `make_group_split.py` (this folder). Logs: `logs/`. Figures and tables: `outputs/`. Splits: `splits/`.

## Finding
The official SOS test sets are **not independent of training**.

**Structure.**
- Sentinel-1 tiles 20001–22220 are 2,220 = **5 × 444** tiles.
- PALSAR tiles 10001–12545 are 2,545 = **5 × 509** tiles.
- Each block is a separately augmented version (crop, rotation, flip, zoom, noise) of the same 444 (or 509) base sea regions.
- Tile *i* and tile *i* + 444·k (PALSAR: + 509·k) show the same region.

**Split.**
- The official test set is simply the first IDs:
  - Sentinel-1: 20001–20839;
  - PALSAR: 10001–10776.
- So **100% of test tiles (839/839 and 776/776) have at least 2 augmented siblings in training**.

## Evidence
1. **Matches fall on the predicted offsets.** For each test tile we searched all training tiles for the best matching central region (8 rotations and flips, normalised cross-correlation).
   - The single best match fell exactly on a sibling offset (+444/+888/+1332, or +509/+1018/+1527) for **571/839 Sentinel-1 and 538/776 PALSAR** test tiles.
   - Chance would be about 0.1%.
2. **Exact copies.**
   - Region score ≥ 0.99, meaning pixel-identical speckle: Sentinel-1 20 tiles (2.4%), PALSAR 34 (4.4%).
   - Above the null threshold (0.944): 109/839 (13.0%) and 149/776 (19.2%).
   - The null threshold is the 99.9th percentile of Sentinel-1 test tiles against PALSAR training tiles.
   - These counts **understate** leakage. Augmentations such as zoom, crop and noise lower the score while the content stays the same.
3. **Visual confirmation.**
   - `sibling_sets.png`: same region in all 5 versions.
   - `check_aligned_sentinel.png`: aligned windows, with full-resolution NCC of 0.97–1.00 for exact copies.
   - `check_pairs_sentinel.png`: score tiers.

## Unknowns
- Training tiles 30001–31973 (Sentinel-1) and 31974–33305 (PALSAR) are in a second block, and their link to base regions is not established.
- The base regions are crops, not the 21 original scenes. Neighbouring base regions can still come from one scene.

## Use
- **C1:** `sos_sibling_groups.csv` gives `base_group` per tile. A group-disjoint split, with no base region in both train and test, is the minimum leakage-free protocol for SOS. A scene-level split needs the scene list from the authors (zhuqq@cug.edu.cn).
- **Run A (official split):** scores remain valid for *reproduction*, but they measure performance on augmented copies of training regions.

## Update: block 2 and group-disjoint split
- **Sentinel-1 block 2** (30001–31973) holds **3 augmented versions of about 661 further regions**: the best match falls at offset 661 or 1,322 for 1,378 of 1,973 tiles. Only 20 tiles match block-1 regions, and those groups were merged.
- **PALSAR block 2** shows no clear period. Each tile is its own group, except 113 tiles matched to block-1 groups.
- **Splits** (`make_group_split.py`, seed 0): `splits/sos_<sensor>_group_s0.csv`.
  - No group appears in more than one partition (verified).
  - Sentinel-1: train 2,516 / val 422 / calib 420 / test 835 tiles (1,089 groups).
  - PALSAR: train 2,327 / val 391 / calib 384 / test 775.
  - The new test set is close in size to the official one (839 / 776), so scores are directly comparable.
