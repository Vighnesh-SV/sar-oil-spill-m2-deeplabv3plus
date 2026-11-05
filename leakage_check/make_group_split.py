"""C1: build group-disjoint SOS splits (no base region in more than one partition).

Groups:
  block 1 (S1 20001-22220, PALSAR 10001-12545): base_group = (id - start) mod 444 / 509 (verified)
  S1 block 2 (30001-31973): 3 augmented versions of ~661 regions -> group (id - 30001) mod 661
  PALSAR block 2: no periodic structure found -> each tile its own group
  any block-2 group with a member matched to a block-1 group is merged into that group
Partitions by group (seeded): train 60 % / val 10 % (temperature, C2) / calib 10 % (conformal, C3) /
test 20 %.  Output: splits/sos_<sensor>_group_s<seed>.csv  (tile_id, sat, mask, partition, group)
"""
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "common"))
sys.path.insert(0, HERE)
from sos_data import list_split  # noqa: E402

FRAC = (("train", 0.6), ("val", 0.1), ("calib", 0.1), ("test", 0.2))


def main(seed=0):
    rows = list(csv.DictReader(open(os.path.join(HERE, "outputs", "sos_sibling_groups.csv"))))
    os.makedirs(os.path.join(HERE, "splits"), exist_ok=True)
    report = {}
    for s, b2start, period in (("sentinel", 30001, 661), ("palsar", 31974, None)):
        R = {r["tile_id"]: r for r in rows if r["sensor"] == s}
        grp = {}
        for t, r in R.items():
            if r["assigned_by"] == "id_block":
                grp[t] = r["base_group"]
            elif period:
                grp[t] = f"{s[0]}B2_{(int(t) - b2start) % period:03d}"
            else:
                grp[t] = f"{s[0]}B2_{t}"
        # merge block-2 groups into block-1 groups where any member matched
        merge = {}
        for t, r in R.items():
            if r["assigned_by"] in ("region", "copy"):
                merge[grp[t]] = r["base_group"]
        grp = {t: merge.get(g, g) for t, g in grp.items()}

        items = {i[0]: i for i in list_split("train", s) + list_split("test", s)}
        gids = sorted(set(grp.values()))
        rng = np.random.default_rng(seed)
        rng.shuffle(gids)
        # assign partitions by cumulative tile count so fractions hold in tiles, not groups
        size = collections.Counter(grp.values())
        total, acc, part = sum(size.values()), 0, {}
        bounds = np.cumsum([f for _, f in FRAC])
        for g in gids:
            frac = acc / total
            part[g] = FRAC[int(np.searchsorted(bounds, frac, side="right"))][0]
            acc += size[g]
        out = os.path.join(HERE, "splits", f"sos_{s}_group_s{seed}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["tile_id", "sat", "mask", "partition", "group"])
            for t in sorted(grp):
                w.writerow([t, items[t][1], items[t][2], part[grp[t]], grp[t]])
        cnt = collections.Counter(part[grp[t]] for t in grp)
        report[s] = {"groups": len(gids), "tiles": dict(cnt),
                     "merged_block2_groups": len(merge),
                     "official_test_tiles_now_in": dict(collections.Counter(
                         part[grp[t]] for t in grp if R[t]["split"] == "test"))}
        print(s, report[s])
    json.dump(report, open(os.path.join(HERE, "splits", f"sos_group_s{seed}_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
