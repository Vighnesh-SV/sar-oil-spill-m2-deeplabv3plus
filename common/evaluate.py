"""Evaluate a trained segmenter on an SOS test split.

* Prediction: official CBD-Net test-time augmentation (8 dihedral transforms, mean prob,
  threshold 0.5 == official 'sum > 4') and a single plain pass (no TTA).
* Saves: probability maps (float16 .npz) for later calibration / conformal work (C2, C3),
  binary masks as PNG (official format, 255 = oil) and a qualitative figure.
* Metrics (dataset-level, pixels pooled over all tiles, as in the official t1p3-Iou.py):
    accuracy, oil precision / recall / F1 / IoU, background IoU, mIoU,
    background-class P/R/F1 (what the official script labels 'fscore'),
    boundary F1 (2-px tolerance), ROC-AUC, PR-AUC, ECE (15 bins), Brier score.

usage (from experiments/):  python common/evaluate.py --run runs/M1_cbdnet_sentinel_dicebce_s0 --sensor sentinel
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "third_party", "CBDNet-main"))
sys.path.insert(0, HERE)
os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "torch"))

from sos_data import list_from_split_csv, list_split, load_eval, normalise  # noqa: E402

NBINS_HIST = 1000


def build_model(arch):
    from models import build_model as _b
    return _b(arch, pretrained=False)  # weights come from the checkpoint


@torch.no_grad()
def predict(net, imgs, dev):
    """imgs: list of HxWx3 uint8 BGR. Returns (prob_tta, prob_plain) arrays [N,H,W]."""
    x = torch.from_numpy(np.stack([normalise(i) for i in imgs])).to(dev)
    outs = []
    for k in range(4):
        for flip in (False, True):
            t = torch.rot90(x, k, dims=(2, 3))
            if flip:
                t = torch.flip(t, dims=(3,))
            p = net(t)
            if flip:
                p = torch.flip(p, dims=(3,))
            outs.append(torch.rot90(p, -k, dims=(2, 3)))
    plain = outs[0][:, 0]
    tta = torch.stack(outs).mean(0)[:, 0]
    return tta.float().cpu().numpy(), plain.float().cpu().numpy()


def boundary(mask):
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


class Acc:
    """Pooled accumulators for all metrics."""

    def __init__(self):
        self.cm = np.zeros(4, np.int64)               # TP FP FN TN (oil = positive)
        self.bcnt = np.zeros(4, np.int64)             # pred-bnd matched, pred-bnd, gt-bnd matched, gt-bnd
        self.hpos = np.zeros(NBINS_HIST, np.int64)    # prob histograms for AUC
        self.hneg = np.zeros(NBINS_HIST, np.int64)
        self.ece_n = np.zeros(15); self.ece_conf = np.zeros(15); self.ece_acc = np.zeros(15)
        self.brier = 0.0; self.n = 0
        self.per_tile = []

    def add(self, tid, prob, gt, tol=2):
        pred = prob > 0.5
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        tn = gt.size - tp - fp - fn
        self.cm += [tp, fp, fn, tn]
        pb, gb = boundary(pred), boundary(gt)
        k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
        gbd = cv2.dilate(gb.astype(np.uint8), k) > 0
        pbd = cv2.dilate(pb.astype(np.uint8), k) > 0
        self.bcnt += [int((pb & gbd).sum()), int(pb.sum()), int((gb & pbd).sum()), int(gb.sum())]
        b = np.minimum((prob * NBINS_HIST).astype(int), NBINS_HIST - 1)
        self.hpos += np.bincount(b[gt], minlength=NBINS_HIST)
        self.hneg += np.bincount(b[~gt], minlength=NBINS_HIST)
        conf = np.maximum(prob, 1 - prob); correct = (pred == gt)
        eb = np.minimum(((conf - 0.5) / 0.5 * 15).astype(int), 14)
        self.ece_n += np.bincount(eb.ravel(), minlength=15)
        self.ece_conf += np.bincount(eb.ravel(), weights=conf.ravel(), minlength=15)
        self.ece_acc += np.bincount(eb.ravel(), weights=correct.ravel(), minlength=15)
        self.brier += float(((prob - gt) ** 2).sum()); self.n += gt.size
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
        self.per_tile.append((tid, iou, int(gt.sum()), int(pred.sum())))

    def result(self):
        tp, fp, fn, tn = [float(v) for v in self.cm]
        d = lambda a, b: a / b if b else float("nan")  # noqa: E731
        P, R = d(tp, tp + fp), d(tp, tp + fn)
        iou_oil, iou_bg = d(tp, tp + fp + fn), d(tn, tn + fp + fn)
        Pb, Rb = d(tn, tn + fn), d(tn, tn + fp)
        bp, br = d(self.bcnt[0], self.bcnt[1]), d(self.bcnt[2], self.bcnt[3])
        # ROC-AUC / PR-AUC from histograms (threshold sweep high -> low)
        tps = np.cumsum(self.hpos[::-1]); fps = np.cumsum(self.hneg[::-1])
        tpr = np.r_[0, tps / tps[-1]]; fpr = np.r_[0, fps / fps[-1]]
        roc = float(np.trapezoid(tpr, fpr))
        prec = np.r_[1, tps / np.maximum(tps + fps, 1)]
        pr_auc = float(np.sum(np.diff(tpr) * prec[1:]))
        nz = self.ece_n > 0
        ece = float(np.sum(np.abs(self.ece_acc[nz] - self.ece_conf[nz])) / self.n)
        return {
            "accuracy": (tp + tn) / (tp + fp + fn + tn),
            "precision_oil": P, "recall_oil": R, "f1_oil": d(2 * P * R, P + R),
            "iou_oil": iou_oil, "iou_bg": iou_bg, "miou": (iou_oil + iou_bg) / 2,
            "precision_bg": Pb, "recall_bg": Rb, "f1_bg_officialscript": d(2 * Pb * Rb, Pb + Rb),
            "boundary_f1": d(2 * bp * br, bp + br),
            "roc_auc": roc, "pr_auc": pr_auc, "ece": ece, "brier": self.brier / self.n,
            "oil_pixel_fraction": (tp + fn) / (tp + fp + fn + tn),
        }


def overlay(img, gt, pred):
    g = img[..., 0].astype(np.float32)
    o = np.stack([g, g, g], -1)
    col = {"tp": (0, 200, 0), "fp": (230, 40, 40), "fn": (40, 120, 255)}  # RGB
    for key, m in (("tp", pred & gt), ("fp", pred & ~gt), ("fn", ~pred & gt)):
        o[m] = 0.45 * o[m] + 0.55 * np.array(col[key], np.float32)
    return o.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sensor", default="sentinel", choices=["sentinel", "palsar"])
    ap.add_argument("--arch", default="cbdnet")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: evaluate only N test tiles")
    ap.add_argument("--split", default="", help="group-split CSV; evaluates its 'test' partition")
    a = ap.parse_args()
    run = a.run if os.path.isabs(a.run) else os.path.join(EXP, a.run)
    out = run if not a.limit else os.path.join(run, "_debug")  # debug runs never overwrite results
    os.makedirs(out, exist_ok=True)

    dev = torch.device(a.device)
    net = build_model(a.arch).to(dev)
    net.load_state_dict(torch.load(os.path.join(run, "best.pth"), map_location=dev))
    net.eval()

    items = (list_from_split_csv(os.path.join(EXP, a.split), ["test"]) if a.split
             else list_split("test", a.sensor))[: a.limit or None]
    tag = a.sensor + ("_groupsplit" if a.split else "")   # file suffix and test-set name
    pred_dir = os.path.join(out, f"pred_{tag}"); os.makedirs(pred_dir, exist_ok=True)
    acc_tta, acc_plain = Acc(), Acc()
    probs_tta, probs_plain, ids = [], [], []
    for s in range(0, len(items), a.bs):
        chunk = items[s:s + a.bs]
        loaded = [load_eval(sp, mp) for _, sp, mp in chunk]
        pt, pp = predict(net, [im for im, _ in loaded], dev)
        for (tid, _, _), (im, gt), t, p in zip(chunk, loaded, pt, pp):
            acc_tta.add(tid, t, gt); acc_plain.add(tid, p, gt)
            probs_tta.append(t.astype(np.float16)); probs_plain.append(p.astype(np.float16)); ids.append(tid)
            m = ((t > 0.5) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(pred_dir, f"{tid}_mask.png"), np.dstack([m, m, m]))
    np.savez_compressed(os.path.join(out, f"probs_{tag}.npz"), ids=np.array(ids),
                        tta=np.stack(probs_tta), plain=np.stack(probs_plain))

    res = {"tta": acc_tta.result(), "plain": acc_plain.result(),
           "n_tiles": len(items), "params_M": sum(p.numel() for p in net.parameters()) / 1e6}
    with open(os.path.join(out, f"metrics_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out, f"per_tile_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "iou_oil", "gt_oil_px", "pred_oil_px"]); w.writerows(acc_tta.per_tile)

    # summary table across runs
    summ = os.path.join(EXP, "results", "summary.csv" if not a.limit else "summary_debug.csv")
    os.makedirs(os.path.dirname(summ), exist_ok=True)
    new = not os.path.exists(summ)
    with open(summ, "a", newline="") as f:
        w = csv.writer(f)
        keys = ["accuracy", "precision_oil", "recall_oil", "f1_oil", "iou_oil", "iou_bg", "miou",
                "f1_bg_officialscript", "boundary_f1", "roc_auc", "pr_auc", "ece", "brier"]
        if new:
            w.writerow(["run", "test_set", "mode"] + keys)
        for mode in ("tta", "plain"):
            w.writerow([os.path.basename(run), tag, mode] + [f"{res[mode][k]:.4f}" for k in keys])

    # qualitative figure: 4 median tiles + 4 worst tiles (by oil IoU, tiles with >=1% oil)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pt = [r for r in acc_tta.per_tile if r[2] >= 0.01 * 256 * 256]
    pt.sort(key=lambda r: r[1])
    mid = len(pt) // 2
    pick = [("median", r) for r in pt[mid - 2: mid + 2]] + [("worst", r) for r in pt[:4]]
    lookup = {i[0]: i for i in items}
    fig, ax = plt.subplots(3, len(pick), figsize=(2.1 * len(pick), 6.6))
    probs_by_id = dict(zip(ids, probs_tta))
    for j, (kind, (tid, iou, _, _)) in enumerate(pick):
        im, gt = load_eval(lookup[tid][1], lookup[tid][2])
        pr = probs_by_id[tid].astype(np.float32) > 0.5
        ax[0, j].imshow(im[..., ::-1]); ax[0, j].set_title(f"{kind} #{tid}\nIoU {iou:.2f}", fontsize=8)
        ax[1, j].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[2, j].imshow(overlay(im, gt, pr))
        for r in range(3):
            ax[r, j].axis("off")
    ax[0, 0].text(-30, 128, "SAR", rotation=90, va="center", fontsize=9)
    ax[1, 0].text(-30, 128, "Ground truth", rotation=90, va="center", fontsize=9)
    ax[2, 0].text(-30, 128, "TP / FP / FN", rotation=90, va="center", fontsize=9)
    fig.suptitle(f"{os.path.basename(run)} on SOS {tag} test  "
                 f"(green TP, red FP, blue FN)", fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out, f"qualitative_{tag}.png"), dpi=110); plt.close(fig)

    for mode in ("tta", "plain"):
        r = res[mode]
        print(f"[{mode}] acc {r['accuracy']:.4f}  P {r['precision_oil']:.4f}  R {r['recall_oil']:.4f}  "
              f"F1 {r['f1_oil']:.4f}  IoU_oil {r['iou_oil']:.4f}  mIoU {r['miou']:.4f}  "
              f"F1_bg(official) {r['f1_bg_officialscript']:.4f}  BF1 {r['boundary_f1']:.4f}  "
              f"AUC {r['roc_auc']:.4f}  ECE {r['ece']:.4f}  Brier {r['brier']:.4f}")


if __name__ == "__main__":
    main()
