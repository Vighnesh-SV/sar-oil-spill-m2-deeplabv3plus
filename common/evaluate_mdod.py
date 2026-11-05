"""External test on MDOD [N05]: false alerts per look-alike class (frozen model, box labels only).

MDOD (OSF release): 4,673 full Sentinel-1 VV scenes (about 2,550 x 1,930 px, greyscale, land masked
white) with 24,940 boxes. category_id: 1 biogenic slick, 2 low wind, 3 oil spill, 4 rain cell, 5 sea ice
(confirmed from the paper's class order and Fig. 3; see datasets/README.md).

There are no pixel masks, so the test is box-level:
* For each box, the model is run (single pass by default, --tta for 8-way; 256 x 256 tiles, overlaps averaged) on a window around
  the box: the box itself, enlarged to at least 256 x 256 for context.
* Box scores: mean oil probability in the box, and PEAK = 99th-percentile probability (thin oil streaks
  fill only a small part of their box, so the mean dilutes them). Coverage = share of box pixels with p > 0.5.
* A box raises an ALERT if coverage >= 0.02 (any real oil area; primary), also reported at 0.1, 0.2 and 0.5.
    oil boxes:        alert rate = box-level detection rate
    look-alike boxes: alert rate = false-alert rate per class   <- the key number (review Table 7, C4)
* Threshold-free: ROC-AUC of the box score, oil vs each look-alike class (how well the model separates them).
A seeded random sample of up to N boxes per class is used (default 400), to keep the run short.

usage (from experiments/):  python common/evaluate_mdod.py --run runs/M1_cbdnet_sentinel_dicebce_s0
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evaluate import build_model, predict  # noqa: E402

MDOD = os.environ.get("MDOD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "MDOD", "Dataset"))
CLASSES = {1: "biogenic slick", 2: "low wind", 3: "oil spill", 4: "rain cell", 5: "sea ice"}
ALERT_COVERAGE = (0.02, 0.1, 0.2, 0.5)


def load_boxes():
    s = open(os.path.join(MDOD, "label", "label.txt")).read()
    pat = r"id:(\d+),image_id:(\d+),category_id:(\d+),bbox:([\d.eE-]+),([\d.eE-]+),([\d.eE-]+),([\d.eE-]+)"
    return [dict(box_id=int(a), image_id=int(b), cat=int(c), bbox=tuple(map(float, (d, e, f, g))))
            for a, b, c, d, e, f, g in re.findall(pat, s)]


TTA = False  # set by --tta; MDOD default is a single pass (8x faster; TTA changed SOS scores by ~0.3 pp)


@torch.no_grad()
def plain_predict(net, tiles, dev):
    from sos_data import normalise
    x = torch.from_numpy(np.stack([normalise(t) for t in tiles])).to(dev)
    return net(x)[:, 0].float().cpu().numpy()


def predict_window(net, gray, dev):
    """Sliding 256x256 tiles (last tile aligned to the edge), overlaps averaged; TTA optional."""
    H, W = gray.shape
    ys = sorted(set(list(range(0, max(H - 256, 0) + 1, 256)) + [max(H - 256, 0)]))
    xs = sorted(set(list(range(0, max(W - 256, 0) + 1, 256)) + [max(W - 256, 0)]))
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    tiles, pos = [], []
    for y in ys:
        for x in xs:
            t = gray[y:y + 256, x:x + 256]
            tiles.append(np.repeat(t[..., None], 3, 2)); pos.append((y, x))
    bs = 4 if TTA else 16
    for s in range(0, len(tiles), bs):
        p = predict(net, tiles[s:s + bs], dev)[0] if TTA else plain_predict(net, tiles[s:s + bs], dev)
        for (y, x), q in zip(pos[s:s + bs], p):
            acc[y:y + 256, x:x + 256] += q; cnt[y:y + 256, x:x + 256] += 1
    return acc / np.maximum(cnt, 1)


def auc(pos, neg):
    """ROC-AUC = P(score_pos > score_neg), ties count half (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, cnts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnts)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arch", default="cbdnet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per_class", type=int, default=400)
    ap.add_argument("--tta", action="store_true", help="8-way TTA (8x slower)")
    ap.add_argument("--min_side", type=int, default=8, help="skip boxes smaller than this (px)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    global TTA
    TTA = a.tta
    run = a.run if os.path.isabs(a.run) else os.path.join(EXP, a.run)
    debug = a.per_class < 50
    out = os.path.join(run, "mdod" if not debug else "mdod_debug"); os.makedirs(out, exist_ok=True)

    dev = torch.device(a.device)
    net = build_model(a.arch).to(dev)
    net.load_state_dict(torch.load(os.path.join(run, "best.pth"), map_location=dev)); net.eval()

    boxes = load_boxes()
    sizes = {}
    rng = random.Random(a.seed)
    sample = []
    for c in CLASSES:
        cand = [b for b in boxes if b["cat"] == c]
        rng.shuffle(cand); sample += cand[: a.per_class * 2]          # oversample, filter by size below
    by_img = collections.defaultdict(list)
    for b in sample:
        by_img[b["image_id"]].append(b)
    kept = collections.Counter(); rows = []
    for n, (img_id, bs) in enumerate(sorted(by_img.items())):
        gray = np.asarray(Image.open(os.path.join(MDOD, "image", f"{img_id}.png")).convert("L"))
        H, W = gray.shape
        for b in bs:
            if kept[b["cat"]] >= a.per_class:
                continue
            x, y, w, h = b["bbox"]
            x0, y0 = int(round(x * W)), int(round(y * H)); x1, y1 = int(round((x + w) * W)), int(round((y + h) * H))
            if x1 - x0 < a.min_side or y1 - y0 < a.min_side:
                continue
            # window = box enlarged to >= 256 in each direction, clipped to the image
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ww, wh = max(x1 - x0, 256), max(y1 - y0, 256)
            wx0 = int(max(0, min(cx - ww / 2, W - ww))); wy0 = int(max(0, min(cy - wh / 2, H - wh)))
            win = gray[wy0:wy0 + int(wh), wx0:wx0 + int(ww)]
            if win.shape[0] < 256 or win.shape[1] < 256:
                win = np.pad(win, ((0, max(0, 256 - win.shape[0])), (0, max(0, 256 - win.shape[1]))), mode="reflect")
            prob = predict_window(net, win, dev)
            pb = prob[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0]
            valid = gray[y0:y1, x0:x1] < 255                          # ignore masked land (pure white)
            if valid.sum() < 16:
                continue
            score = float(pb[valid].mean()); cov = float((pb[valid] > 0.5).mean())
            peak = float(np.percentile(pb[valid], 99))
            kept[b["cat"]] += 1
            rows.append([b["box_id"], img_id, b["cat"], CLASSES[b["cat"]], x1 - x0, y1 - y0, round(score, 4), round(cov, 4), round(peak, 4)])
        if n % 200 == 0:
            print(f"images {n}/{len(by_img)}  boxes kept {dict(kept)}", flush=True)
        if all(kept[c] >= a.per_class for c in CLASSES):
            break

    res = {"settings": {"tta": a.tta, "per_class": a.per_class, "min_side": a.min_side, "seed": a.seed,
                        "alert_rule": "coverage = share of box pixels with p>0.5; alert if coverage >= threshold"},
           "per_class": {}, "auc_oil_vs": {}, "auc_oil_vs_peak": {}}
    sc = {c: [r[6] for r in rows if r[2] == c] for c in CLASSES}
    pk = {c: [r[8] for r in rows if r[2] == c] for c in CLASSES}
    cv = {c: np.array([r[7] for r in rows if r[2] == c]) for c in CLASSES}
    for c, name in CLASSES.items():
        res["per_class"][name] = {"n_boxes": len(sc[c]), "mean_box_score": float(np.mean(sc[c])) if sc[c] else None,
                                  "median_coverage": float(np.median(cv[c])) if len(cv[c]) else None,
                                  "mean_peak_score": float(np.mean(pk[c])) if pk[c] else None,
                                  **{f"alert_rate@{t}": float((cv[c] >= t).mean()) if len(cv[c]) else None
                                     for t in ALERT_COVERAGE}}
        if c != 3:
            res["auc_oil_vs"][name] = auc(sc[3], sc[c])
            res["auc_oil_vs_peak"][name] = auc(pk[3], pk[c])
    res["auc_oil_vs"]["all look-alikes"] = auc(sc[3], sum((sc[c] for c in CLASSES if c != 3), []))
    res["auc_oil_vs_peak"]["all look-alikes"] = auc(pk[3], sum((pk[c] for c in CLASSES if c != 3), []))
    json.dump(res, open(os.path.join(out, "metrics_mdod.json"), "w"), indent=2)
    with open(os.path.join(out, "per_box_mdod.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["box_id", "image_id", "cat_id", "class", "w_px", "h_px", "box_score", "coverage", "peak_score"])
        w.writerows(rows)
    figure(rows, os.path.join(out, "qualitative_mdod.png"), net, dev, os.path.basename(run))
    print("class            n   alert@0.02  alert@0.2   mean  peak   AUC(mean) AUC(peak) oil-vs-class")
    for c, name in CLASSES.items():
        r = res["per_class"][name]
        am = res["auc_oil_vs"].get(name, float("nan")) if c != 3 else float("nan")
        ap_ = res["auc_oil_vs_peak"].get(name, float("nan")) if c != 3 else float("nan")
        print(f"{name:15s} {r['n_boxes']:4d}   {r['alert_rate@0.02']:.3f}      {r['alert_rate@0.2']:.3f}      "
              f"{r['mean_box_score']:.3f} {r['mean_peak_score']:.3f}  {am:.3f}     {ap_:.3f}")
    print("AUC oil vs all look-alikes: mean", round(res["auc_oil_vs"]["all look-alikes"], 3),
          " peak", round(res["auc_oil_vs_peak"]["all look-alikes"], 3))


def figure(rows, path, net, dev, title):
    """Two random boxes per class: SAR crop and the predicted oil probability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = random.Random(1)
    fig, ax = plt.subplots(2, 10, figsize=(20, 4.6))
    for k, c in enumerate(CLASSES):
        cand = [r for r in rows if r[2] == c and r[4] >= 64 and r[5] >= 64] or [r for r in rows if r[2] == c]
        for j, r in enumerate(rng.sample(cand, min(2, len(cand)))):
            gray = np.asarray(Image.open(os.path.join(MDOD, "image", f"{r[1]}.png")).convert("L"))
            box = [b for b in load_boxes_cache() if b["box_id"] == r[0]][0]
            H, W = gray.shape; x, y, w, h = box["bbox"]
            x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
            crop = gray[y0:y1, x0:x1]
            pad = np.pad(crop, ((0, max(0, 256 - crop.shape[0])), (0, max(0, 256 - crop.shape[1]))), mode="reflect")
            prob = predict_window(net, pad, dev)[:crop.shape[0], :crop.shape[1]]
            col = 2 * k + j
            ax[0, col].imshow(crop, cmap="gray", vmin=0, vmax=255)
            ax[0, col].set_title(f"{CLASSES[c]}\ncoverage {r[7]:.2f}", fontsize=8)
            ax[1, col].imshow(prob, cmap="magma", vmin=0, vmax=1)
            for i in range(2):
                ax[i, col].axis("off")
    ax[0, 0].text(-0.15, 0.5, "SAR (box)", rotation=90, va="center", transform=ax[0, 0].transAxes, fontsize=9)
    ax[1, 0].text(-0.15, 0.5, "P(oil)", rotation=90, va="center", transform=ax[1, 0].transAxes, fontsize=9)
    fig.suptitle(f"{title} on MDOD: predicted oil probability inside labelled boxes (bright = oil)", fontsize=9)
    plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


_BOX_CACHE = None


def load_boxes_cache():
    global _BOX_CACHE
    if _BOX_CACHE is None:
        _BOX_CACHE = load_boxes()
    return _BOX_CACHE


if __name__ == "__main__":
    main()
