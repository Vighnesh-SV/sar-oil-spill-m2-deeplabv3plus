"""Assign SOS training tiles outside block 1 (S1 30001-31973, PALSAR 31974-33305) to the
444/509 base groups by best match against block-1 tiles (same two stages and null thresholds as
near_duplicates.py). Updates leakage/sos_sibling_groups.csv with columns match_score, assigned_by.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "common"))
sys.path.insert(0, HERE)
import near_duplicates as N  # noqa: E402
from sos_data import list_split  # noqa: E402

L = os.path.join(HERE, "outputs")
null = json.load(open(os.path.join(L, "summary.json")))["null"]
T1, T2 = null["stage1_p999"], null["stage2_p999"]
rows = list(csv.DictReader(open(os.path.join(L, "sos_sibling_groups.csv"))))
for r in rows:
    r["match_score"], r["assigned_by"] = "", ("id_block" if r["base_group"] else "")
stats = {}
for s in ("sentinel", "palsar"):
    items = list_split("train", s) + list_split("test", s)
    by_id = {i[0]: i for i in items}
    grp = {r["tile_id"]: r for r in rows if r["sensor"] == s}
    ref = [by_id[t] for t, r in grp.items() if r["base_group"]]
    q = [by_id[t] for t, r in grp.items() if not r["base_group"]]
    g_ref, z_ref = N.load(ref); g_q, z_q = N.load(q)
    b1, i1 = N.stage1(z_q, z_ref)
    b2, i2 = N.stage2_gpu(g_q, g_ref)
    n_ok = 0
    for k, item in enumerate(q):
        r = grp[item[0]]
        if b2[k] > T2 or b1[k] > T1:
            j = i2[k] if b2[k] >= b1[k] else i1[k]
            r["base_group"] = grp[ref[j][0]]["base_group"]
            r["assigned_by"] = "region" if b2[k] > T2 else "copy"
            r["match_score"] = f"{max(b1[k], b2[k]):.4f}"; n_ok += 1
        else:
            r["assigned_by"] = "unmatched"; r["match_score"] = f"{max(b1[k], b2[k]):.4f}"
    stats[s] = {"block2_tiles": len(q), "assigned": n_ok, "unmatched": len(q) - n_ok,
                "score_median": float(np.median(np.maximum(b1, b2)))}
    print(s, stats[s], flush=True)
with open(os.path.join(L, "sos_sibling_groups.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(stats, open(os.path.join(L, "block2_assignment.json"), "w"), indent=2)
