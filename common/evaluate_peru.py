"""External test on the Peru dataset [N22] (frozen model, no Peru data used for training).

Peru: 2,013 Sentinel-1 patches (512 x 512, one band copied to 3 channels, the same format as SOS)
from 40 scenes, with 5-class masks: 0 sea, 1 oil, 2 look-alike, 3 ship, 4 land
(1 and 2 are confirmed by the dataset's correction log; 0/3/4 follow the usual convention).

* Each 512 x 512 patch is cut into four 256 x 256 tiles (the training input size), predicted with
  the same 8-way TTA as evaluate.py, and stitched back.
* Binary scoring: oil (class 1) = positive, every other class = negative, with the same pooled
  metrics as the SOS evaluation (Acc class from evaluate.py).
* Extra, Peru-specific: the share of each non-oil class that the model calls oil
  (the look-alike-as-oil rate is the key one) and the oil recall per scene.
* Reported on all 2,013 patches (40 scenes; valid because no Peru data was used in training) and on
  the dataset's official 338-patch test subset (5 scenes) for comparison with the Peru paper.

usage (from experiments/):  python common/evaluate_peru.py --run runs/M1_cbdnet_sentinel_dicebce_s0
"""
import argparse
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
from evaluate import Acc, build_model, overlay, predict  # noqa: E402

PERU = os.environ.get("PERU_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Peru", "peru_event_region_v1_jan25_copy"))
CLASSES = {0: "sea", 1: "oil", 2: "look-alike", 3: "ship", 4: "land"}


def peru_items():
    rows = list(csv.DictReader(open(os.path.join(PERU, "manifests", "patch_manifest.tsv")), delimiter="\t"))
    return [dict(split=r["split"], img=os.path.join(PERU, r["image_path"]), lab=os.path.join(PERU, r["label_path"]),
                 scene=r["scene_id"], date=r["acq_date"], zone=r["coastal_zone"]) for r in rows]


def predict_patch(net, img, dev):
    """512x512 -> four 256x256 tiles -> TTA -> stitched probability map."""
    tiles = [img[y:y + 256, x:x + 256] for y in (0, 256) for x in (0, 256)]
    p, _ = predict(net, tiles, dev)
    out = np.zeros(img.shape[:2], np.float32)
    for k, (y, x) in enumerate([(0, 0), (0, 256), (256, 0), (256, 256)]):
        out[y:y + 256, x:x + 256] = p[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arch", default="cbdnet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run = a.run if os.path.isabs(a.run) else os.path.join(EXP, a.run)
    out = os.path.join(run, "peru" if not a.limit else "peru_debug"); os.makedirs(out, exist_ok=True)

    dev = torch.device(a.device)
    net = build_model(a.arch).to(dev)
    net.load_state_dict(torch.load(os.path.join(run, "best.pth"), map_location=dev)); net.eval()

    items = peru_items()[: a.limit or None]
    acc = {"all": Acc(), "official_test": Acc()}
    cls_tot = {k: np.zeros(5, np.int64) for k in acc}   # pixels per class
    cls_oil = {k: np.zeros(5, np.int64) for k in acc}   # of those, predicted oil
    scene = collections.defaultdict(lambda: np.zeros(3, np.int64))  # oil px, oil found, look-alike px called oil
    probs, rows = [], []
    for n, it in enumerate(items):
        img = cv2.imread(it["img"], cv2.IMREAD_COLOR)            # 3 identical channels, uint8
        lab = cv2.imread(it["lab"], cv2.IMREAD_UNCHANGED)
        prob = predict_patch(net, img, dev)
        pred = prob > 0.5
        gt = lab == 1
        for k in (["all", "official_test"] if it["split"] == "test" else ["all"]):
            acc[k].add(os.path.basename(it["img"]), prob, gt)
            cls_tot[k] += np.bincount(lab.ravel(), minlength=5)[:5]
            cls_oil[k] += np.bincount(lab[pred].ravel(), minlength=5)[:5]
        s = scene[it["scene"]]
        s += [int(gt.sum()), int((gt & pred).sum()), int((pred & (lab == 2)).sum())]
        probs.append((prob * 255).astype(np.uint8))
        rows.append([os.path.basename(it["img"]), it["scene"], it["split"], int(gt.sum()), int(pred.sum()),
                     int((lab == 2).sum()), int((pred & (lab == 2)).sum())])
        if n % 400 == 0:
            print(f"{n}/{len(items)}", flush=True)

    res = {}
    for k in acc:
        r = acc[k].result()
        r["called_oil_rate_per_class"] = {CLASSES[c]: (float(cls_oil[k][c] / cls_tot[k][c]) if cls_tot[k][c] else None)
                                          for c in range(5)}
        r["class_pixel_share"] = {CLASSES[c]: float(cls_tot[k][c] / cls_tot[k].sum()) for c in range(5)}
        r["n_patches"] = len(acc[k].per_tile)
        res[k] = r
    res["per_scene"] = {sc: {"oil_px": int(v[0]), "oil_recall": (float(v[1] / v[0]) if v[0] else None),
                             "lookalike_px_called_oil": int(v[2])} for sc, v in sorted(scene.items())}
    json.dump(res, open(os.path.join(out, "metrics_peru.json"), "w"), indent=2)
    with open(os.path.join(out, "per_patch_peru.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch", "scene", "peru_split", "gt_oil_px", "pred_oil_px", "lookalike_px", "lookalike_px_called_oil"])
        w.writerows(rows)
    np.savez_compressed(os.path.join(out, "probs_peru_uint8.npz"), names=np.array([r[0] for r in rows]),
                        prob=np.stack(probs))

    summ = os.path.join(EXP, "results", "summary.csv" if not a.limit else "summary_debug.csv")
    keys = ["accuracy", "precision_oil", "recall_oil", "f1_oil", "iou_oil", "iou_bg", "miou",
            "f1_bg_officialscript", "boundary_f1", "roc_auc", "pr_auc", "ece", "brier"]
    with open(summ, "a", newline="") as f:
        w = csv.writer(f)
        for k in acc:
            w.writerow([os.path.basename(run), f"peru_{k}", "tta"] + [f"{res[k][m]:.4f}" for m in keys])

    figure(items, rows, probs, os.path.join(out, "qualitative_peru.png"), os.path.basename(run))
    for k in acc:
        r = res[k]
        print(f"[peru {k}: {r['n_patches']} patches] P {r['precision_oil']:.4f}  R {r['recall_oil']:.4f}  "
              f"F1 {r['f1_oil']:.4f}  IoU_oil {r['iou_oil']:.4f}  mIoU {r['miou']:.4f}  ECE {r['ece']:.4f}  "
              f"look-alike->oil {r['called_oil_rate_per_class']['look-alike']:.4f}")


def figure(items, rows, probs, path, title):
    """3 oil patches (median oil IoU) + 3 patches with the most look-alike area."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = np.array([[0, 0, 0], [255, 255, 255], [235, 160, 30], [220, 40, 220], [90, 90, 90]], np.uint8)
    idx = {r[0]: k for k, r in enumerate(rows)}
    oil = [r for r in rows if r[3] > 0.02 * 512 * 512]
    iou = []
    for r in oil:
        k = idx[r[0]]; lab = cv2.imread(items[k]["lab"], cv2.IMREAD_UNCHANGED); p = probs[k] > 127
        g = lab == 1; iou.append(((p & g).sum() / max((p | g).sum(), 1), r))
    iou.sort(key=lambda t: t[0]); mid = len(iou) // 2
    pick = [("oil (median)", t[1]) for t in iou[mid - 1: mid + 2]]
    pick += [("look-alike", r) for r in sorted(rows, key=lambda r: -r[5])[:3]]
    fig, ax = plt.subplots(3, len(pick), figsize=(2.3 * len(pick), 7.2))
    for j, (kind, r) in enumerate(pick):
        k = idx[r[0]]; img = cv2.imread(items[k]["img"]); lab = cv2.imread(items[k]["lab"], cv2.IMREAD_UNCHANGED)
        p = probs[k] > 127
        ax[0, j].imshow(img[..., ::-1]); ax[0, j].set_title(kind, fontsize=8)
        ax[1, j].imshow(colors[lab])
        ax[2, j].imshow(overlay(img, lab == 1, p))
        for i in range(3):
            ax[i, j].axis("off")
    ax[0, 0].text(-40, 256, "SAR", rotation=90, va="center", fontsize=9)
    ax[1, 0].text(-40, 256, "Label", rotation=90, va="center", fontsize=9)
    ax[2, 0].text(-40, 256, "Oil: TP / FP / FN", rotation=90, va="center", fontsize=9)
    fig.suptitle(f"{title} on Peru (labels: white oil, orange look-alike, grey land, magenta ship; "
                 f"overlay: green TP, red FP, blue FN)", fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
