"""Overfitting, dataset and uncertainty analysis for the trained M1 / M2 models (no training).

1. Train-vs-test gap: every trained model is scored (single pass, no augmentation) on its own training tiles and on
   its test tiles, with the same pooled metrics and the training loss (BCE + Dice). A large gap = overfitting.
2. Dataset checks: SOS duplicates (effective number of independent sea areas), oil share per split, Peru scene
   balance, MDOD class sizes and box sizes.
3. 95 % bootstrap confidence intervals (2,000 resamples): SOS by tile, Peru by scene (cluster bootstrap), MDOD by
   box; paired differences (same test items) for M1 vs M2 and for leaky- vs leak-free-trained models.

Writes results/analysis/analysis.json.   usage (from experiments/):  python tools/analysis_overfit.py
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EXP, "common"))
sys.path.insert(0, os.path.join(EXP, "third_party", "CBDNet-main"))
from dice_bce_loss import dice_bce_loss  # noqa: E402
from evaluate import Acc, build_model  # noqa: E402
from evaluate_peru import peru_items  # noqa: E402
from evaluate_mdod import MDOD  # noqa: E402
from sos_data import SOSTrain, list_from_split_csv, list_split  # noqa: E402

OUT = os.path.join(EXP, "results", "analysis"); os.makedirs(OUT, exist_ok=True)
SPLIT = os.path.join(EXP, "leakage_check", "splits", "sos_sentinel_group_s0.csv")
RUNS = {  # key: (run folder, arch, split)
    "M1_official": ("M1_cbdnet_sentinel_dicebce_s0", "cbdnet", "official"),
    "M1_groupsplit": ("M1_cbdnet_sentinel_dicebce_s0_groupsplit", "cbdnet", "group"),
    "M2_official": ("M2_dlv3p_mnv2_sentinel_s0", "dlv3p_mnv2", "official"),
    "M2_groupsplit": ("M2_dlv3p_mnv2_sentinel_s0_groupsplit", "dlv3p_mnv2", "group"),
}
B = 2000
rng = np.random.default_rng(0)
KEYS = ["accuracy", "precision_oil", "recall_oil", "f1_oil", "iou_oil", "iou_bg", "miou", "boundary_f1", "roc_auc", "ece", "brier"]


def items_for(split, part):
    if split == "official":
        return list_split("train" if part == "train" else "test", "sentinel")
    return list_from_split_csv(SPLIT, ["train", "val", "calib"] if part == "train" else ["test"])


@torch.no_grad()
def score(net, items, dev):
    dl = torch.utils.data.DataLoader(SOSTrain(items, augment=False), batch_size=8, shuffle=False, num_workers=2)
    acc, crit, tot, n = Acc(), dice_bce_loss(), 0.0, 0
    for img, mask in dl:
        p = net(img.to(dev))
        tot += float(crit(mask.to(dev), p)) * len(img); n += len(img)
        pn = p.float().cpu().numpy()[:, 0]
        for k in range(len(pn)):
            acc.add(None, pn[k], mask[k, 0].numpy() > 0.5)
    r = acc.result(); r["loss"] = tot / n; r["n_tiles"] = n
    return {k: r[k] for k in KEYS + ["loss", "n_tiles"]}


def train_vs_test():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for key, (run, arch, split) in RUNS.items():
        net = build_model(arch).to(dev)
        net.load_state_dict(torch.load(os.path.join(EXP, "runs", run, "best.pth"), map_location=dev)); net.eval()
        res[key] = {part: score(net, items_for(split, part), dev) for part in ("train", "test")}
        print(key, {p: round(res[key][p]["iou_oil"], 4) for p in res[key]}, flush=True)
        del net; torch.cuda.empty_cache()
    return res


def counts_from_per_tile(path):
    """per-tile TP/FP/FN/TN from (iou, gt px, pred px): TP = iou (g + p) / (1 + iou)."""
    ids, c = [], []
    for r in csv.DictReader(open(path)):
        iou, g, p = float(r["iou_oil"]), int(r["gt_oil_px"]), int(r["pred_oil_px"])
        tp = 0 if g + p == 0 else round(iou * (g + p) / (1 + iou))
        fp, fn = p - tp, g - tp
        ids.append(r["id"]); c.append([tp, fp, fn, 256 * 256 - tp - fp - fn])
    return ids, np.array(c, np.float64)


def pooled(c):
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    P, R = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    iou, iou_bg = tp / np.maximum(tp + fp + fn, 1), tn / np.maximum(tn + fp + fn, 1)
    return dict(precision_oil=P, recall_oil=R, f1_oil=2 * P * R / np.maximum(P + R, 1e-12), iou_oil=iou,
                miou=(iou + iou_bg) / 2)


def boot(c, groups=None, c2=None):
    """Bootstrap pooled metrics; c2 = second model on the same items (paired difference). groups = cluster ids."""
    if groups is None:
        groups = np.arange(len(c))
    ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
    gc = np.stack([c[gi[g]].sum(0) for g in ug]); gc2 = None if c2 is None else np.stack([c2[gi[g]].sum(0) for g in ug])
    idx = rng.integers(0, len(ug), (B, len(ug)))
    s = gc[idx].sum(1); m = pooled(s); point = pooled(gc.sum(0))
    out = {k: dict(value=float(point[k]), lo=float(np.percentile(m[k], 2.5)), hi=float(np.percentile(m[k], 97.5))) for k in m}
    if c2 is not None:
        m2, p2 = pooled(gc2[idx].sum(1)), pooled(gc2.sum(0))
        for k in m:
            d = m2[k] - m[k]
            out[k]["diff_b_minus_a"] = dict(value=float(p2[k] - point[k]), lo=float(np.percentile(d, 2.5)),
                                            hi=float(np.percentile(d, 97.5)), p_b_le_a=float((d <= 0).mean()))
    return out


def sos_ci():
    tiles = {k: counts_from_per_tile(os.path.join(EXP, "runs", r, f"per_tile_sentinel{'' if s == 'official' else '_groupsplit'}.csv"))
             for k, (r, _, s) in RUNS.items()}
    res = {k: boot(c) for k, (_, c) in tiles.items()}
    for a, b in [("M1_official", "M2_official"), ("M1_groupsplit", "M2_groupsplit")]:
        ida, ca = tiles[a]; idb, cb = tiles[b]; pos = {t: i for i, t in enumerate(idb)}
        res[f"{a}_vs_{b}"] = boot(ca, c2=cb[[pos[t] for t in ida]])
    return res


def peru_counts(run):
    z = np.load(os.path.join(EXP, "runs", run, "peru", "probs_peru_uint8.npz"))
    items = {os.path.basename(i["img"]): i for i in peru_items()}
    names, c, scenes, la = [], [], [], []
    for n, p in zip(z["names"], z["prob"]):
        it = items[str(n)]; lab = cv2.imread(it["lab"], cv2.IMREAD_UNCHANGED); pr = p > 127; gt = lab == 1
        tp = int((pr & gt).sum()); fp = int((pr & ~gt).sum()); fn = int((~pr & gt).sum())
        names.append(str(n)); scenes.append(it["scene"]); c.append([tp, fp, fn, gt.size - tp - fp - fn])
        la.append([int((lab == 2).sum()), int(((lab == 2) & pr).sum())])
    return names, np.array(c, np.float64), np.array(scenes), np.array(la, np.float64)


def peru_ci():
    data = {k: peru_counts(r) for k, (r, _, _) in RUNS.items()}
    res = {}
    for k, (names, c, sc, la) in data.items():
        res[k] = boot(c, groups=sc)
        ug = np.unique(sc); gla = np.stack([la[sc == g].sum(0) for g in ug])
        bl = gla[rng.integers(0, len(ug), (B, len(ug)))].sum(1)
        r = bl[:, 1] / np.maximum(bl[:, 0], 1)
        res[k]["lookalike_called_oil"] = dict(value=float(gla[:, 1].sum() / gla[:, 0].sum()), lo=float(np.percentile(r, 2.5)),
                                              hi=float(np.percentile(r, 97.5)))
    for a, b in [("M1_official", "M1_groupsplit"), ("M2_official", "M2_groupsplit"), ("M1_official", "M2_official"),
                 ("M1_groupsplit", "M2_groupsplit")]:
        na, ca, sa, _ = data[a]; nb, cb, _, _ = data[b]; pos = {t: i for i, t in enumerate(nb)}
        res[f"{a}_vs_{b}"] = boot(ca, groups=sa, c2=cb[[pos[t] for t in na]])
    sizes = Counter(data["M1_official"][2])
    return res, dict(n_patches=int(len(data["M1_official"][0])), n_scenes=len(sizes),
                     largest_scene_share=max(sizes.values()) / sum(sizes.values()),
                     patches_per_scene_median=float(np.median(list(sizes.values()))),
                     patches_per_scene_max=max(sizes.values()), patches_per_scene_min=min(sizes.values()))


def auc(pos, neg):
    allv = np.concatenate([pos, neg]); r = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mdod_ci():
    res, sizes = {}, {}
    for k, (r, _, _) in RUNS.items():
        rows = list(csv.DictReader(open(os.path.join(EXP, "runs", r, "mdod", "per_box_mdod.csv"))))
        cls = np.array([x["class"] for x in rows]); s = np.array([float(x["box_score"]) for x in rows])
        pos, neg = s[cls == "oil spill"], s[cls != "oil spill"]
        bs = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(500)]
        res[k] = dict(value=float(auc(pos, neg)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        if not sizes:
            for c in np.unique(cls):
                w = np.array([int(x["w_px"]) for x in rows if x["class"] == c]); h = np.array([int(x["h_px"]) for x in rows if x["class"] == c])
                sizes[str(c)] = dict(sampled=int(len(w)), median_short_side_px=float(np.median(np.minimum(w, h))),
                                     median_area_kpx=float(np.median(w * h) / 1e3))
    # all boxes in the release, per class
    from evaluate_mdod import load_boxes
    tot = Counter(b["cat"] for b in load_boxes())
    names = {1: "biogenic slick", 2: "low wind", 3: "oil spill", 4: "rain cell", 5: "sea ice"}
    for i, c in names.items():
        sizes.setdefault(c, {})["in_release"] = tot.get(i, 0)
    return res, sizes


def sos_dataset():
    sp = list(csv.DictReader(open(SPLIT)))   # group column = independent sea area, covers all 4,193 tiles
    grp = {r["tile_id"]: r["group"] for r in sp}
    by = {"train": {grp[i] for i, _, _ in list_split("train", "sentinel")},
          "test": {grp[i] for i, _, _ in list_split("test", "sentinel")}}
    part = Counter(r["partition"] for r in sp); pg = defaultdict(set)
    for r in sp:
        pg[r["partition"]].add(r["group"])

    def oil_share(items):
        tot = s = 0
        for _, _, m in items:
            mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE) > 127; s += int(mk.sum()); tot += mk.size
        return s / tot
    return dict(official=dict(train_tiles=len(list_split("train", "sentinel")), test_tiles=len(list_split("test", "sentinel")),
                              train_groups=len(by["train"]), test_groups=len(by["test"]),
                              shared_groups=len(by["train"] & by["test"]),
                              oil_share_train=oil_share(list_split("train", "sentinel")),
                              oil_share_test=oil_share(list_split("test", "sentinel"))),
                group=dict(tiles=dict(part), groups={k: len(v) for k, v in pg.items()},
                           shared_groups_train_test=len((pg["train"] | pg["val"] | pg["calib"]) & pg["test"]),
                           oil_share_train=oil_share(list_from_split_csv(SPLIT, ["train", "val", "calib"])),
                           oil_share_test=oil_share(list_from_split_csv(SPLIT, ["test"]))),
                total_groups=len(by["train"] | by["test"]))


if __name__ == "__main__":
    A = {}
    A["sos_dataset"] = sos_dataset(); print("dataset done", flush=True)
    A["sos_ci"] = sos_ci(); print("sos ci done", flush=True)
    A["peru_ci"], A["peru_dataset"] = peru_ci(); print("peru ci done", flush=True)
    A["mdod_auc_ci"], A["mdod_dataset"] = mdod_ci(); print("mdod done", flush=True)
    A["train_vs_test"] = train_vs_test()
    json.dump(A, open(os.path.join(OUT, "analysis.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "analysis.json"))
