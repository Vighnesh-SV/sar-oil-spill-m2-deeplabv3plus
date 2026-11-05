"""Build the presentable M2 results folders from the finished runs.

Writes M2_DeepLabV3Plus/results/{sentinel_official_split, sentinel_group_split, peru_external, mdod_external}/
with results.md, figures and the metric files.
usage (from experiments/):  python M2_DeepLabV3Plus/make_report_m2.py
"""
import csv
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
RUN_O = os.path.join(EXP, "runs", "M2_dlv3p_mnv2_sentinel_s0")
RUN_G = os.path.join(EXP, "runs", "M2_dlv3p_mnv2_sentinel_s0_groupsplit")
M1_O = os.path.join(EXP, "runs", "M1_cbdnet_sentinel_dicebce_s0")
M1_G = os.path.join(EXP, "runs", "M1_cbdnet_sentinel_dicebce_s0_groupsplit")
PUBLISHED = {"miou": 0.8134, "f1_oil": 0.8962, "src": "Zhang et al. 2024 [P01], SOS Sentinel-1"}
SERIES, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
CLASSES = ["oil spill", "biogenic slick", "low wind", "rain cell", "sea ice"]
LA = CLASSES[1:]
J = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
pc = lambda v: f"{100 * v:.2f}"  # noqa: E731
p1 = lambda v: f"{100 * v:.1f}"  # noqa: E731
dpp = lambda a, b: f"{100 * (b - a):+.2f}"  # noqa: E731


def training_curve(log_csv, out_png, title):
    rows = list(csv.DictReader(open(log_csv)))
    ep = np.array([int(r["epoch"]) for r in rows]); loss = np.array([float(r["train_loss"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ep, loss, color=SERIES, lw=2)
    ax.scatter([ep[-1]], [loss[-1]], s=40, color=SERIES, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"final {loss[-1]:.3f} (epoch {ep[-1]}, evaluated)", (ep[-1], loss[-1]), xytext=(-8, 14),
                textcoords="offset points", ha="right", color=INK, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("epoch  (learning rate: cosine decay 1e-3 → 0)"); ax.set_ylabel("training loss (BCE + Dice)")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return int(ep[-1]), float(loss[-1]), int(rows[-1]["seconds"])


def per_tile_hist(per_tile_csv, out_png):
    rows = [r for r in csv.DictReader(open(per_tile_csv)) if int(r["gt_oil_px"]) > 0]
    iou = np.array([float(r["iou_oil"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(iou, bins=np.linspace(0, 1, 21), color=SERIES, edgecolor="white", linewidth=2)
    med = float(np.median(iou)); ax.axvline(med, color=INK, lw=1)
    ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.2f}", color=INK, fontsize=8, va="top")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("oil IoU of one test tile"); ax.set_ylabel("number of tiles")
    ax.set_title(f"M2: per-tile oil IoU ({len(iou)} test tiles that contain oil)", loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return med, float((iou < 0.3).mean())


def alert_chart(M, path):
    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    for i, (tag, name, col) in enumerate([("official", "Official-split model", INK2), ("groupsplit", "Leak-free model", SERIES)]):
        v = [100 * M[tag]["per_class"][c]["alert_rate@0.02"] for c in CLASSES]
        bars = ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=col, label=name)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, ["oil spill\n(should alert)"] + LA, fontsize=8.5, color=INK)
    ax.set_ylabel("boxes flagged as oil (%)", fontsize=9, color=INK2)
    ax.set_ylim(0, 110); ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(INK2); ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


SEG = [("Accuracy", "accuracy"), ("Precision (oil)", "precision_oil"), ("Recall (oil)", "recall_oil"),
       ("**F1 (oil)**", "f1_oil"), ("IoU (oil)", "iou_oil"), ("IoU (background)", "iou_bg"), ("**mIoU**", "miou"),
       ("Boundary F1 (2 px)", "boundary_f1"), ("F1 as computed by the official script (background class)", "f1_bg_officialscript")]


def sos_official():
    out = os.path.join(RES, "sentinel_official_split"); os.makedirs(out, exist_ok=True)
    m = J(os.path.join(RUN_O, "metrics_sentinel.json")); m1 = J(os.path.join(M1_O, "metrics_sentinel.json"))["tta"]
    shutil.copy(os.path.join(RUN_O, "metrics_sentinel.json"), os.path.join(out, "metrics.json"))
    shutil.copy(os.path.join(RUN_O, "per_tile_sentinel.csv"), os.path.join(out, "per_tile.csv"))
    shutil.copy(os.path.join(RUN_O, "train_log.csv"), os.path.join(out, "train_log.csv"))
    shutil.copy(os.path.join(RUN_O, "qualitative_sentinel.png"), os.path.join(out, "qualitative.png"))
    ep, loss, sec = training_curve(os.path.join(RUN_O, "train_log.csv"), os.path.join(out, "training_curve.png"),
                                   "M2 Improved DeepLabV3+: training loss on SOS Sentinel-1")
    med, bad = per_tile_hist(os.path.join(RUN_O, "per_tile_sentinel.csv"), os.path.join(out, "per_tile_iou.png"))
    t, p = m["tta"], m["plain"]
    L = ["# M2 Improved DeepLabV3+: results on SOS Sentinel-1 (official split)", "",
         f"Run: `runs/M2_dlv3p_mnv2_sentinel_s0`. Test tiles: {m['n_tiles']}. Parameters: {m['params_M']:.2f} M. "
         f"Trained {ep} epochs ({sec // 60} min); the final-epoch model is evaluated (final loss {loss:.4f}).", "",
         "## Quantitative results (%, pixels pooled over all test tiles)", "",
         "| Metric | Ours, with TTA | Ours, single pass | Published [P01] | M1 CBD-Net (ours, TTA) |", "|---|---|---|---|---|"]
    for lab, k in SEG:
        pub = pc(PUBLISHED[k]) if k in PUBLISHED else "—"
        L.append(f"| {lab} | {pc(t[k])} | {pc(p[k])} | {pub} | {pc(m1[k])} |")
    L += ["", "| Probability quality | With TTA | Single pass | M1 (TTA) |", "|---|---|---|---|"]
    for lab, k in [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("ECE (15 bins)", "ece"), ("Brier score", "brier")]:
        L.append(f"| {lab} | {t[k]:.4f} | {p[k]:.4f} | {m1[k]:.4f} |")
    L += ["", f"**Reproduction gap:** mIoU {dpp(PUBLISHED['miou'], t['miou'])} pp, F1 {dpp(PUBLISHED['f1_oil'], t['f1_oil'])} pp "
          "(target: within ±2 pp). The paper does not say which class its F1 is computed for; if it used the official "
          f"SOS script (background class), the comparable value is {pc(t['f1_bg_officialscript'])} %.", "",
          "## Figures", "- `training_curve.png`: training loss per epoch.",
          f"- `per_tile_iou.png`: per-tile oil IoU. Median {med:.2f}; {100 * bad:.0f} % of oil tiles score below 0.3.",
          "- `qualitative.png`: four median and four worst test tiles (green = correct oil, red = false oil, blue = missed oil).", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(L))
    return dict(epochs=ep, loss=loss, sec=sec, med=med, bad=bad)


def sos_group():
    out = os.path.join(RES, "sentinel_group_split"); os.makedirs(out, exist_ok=True)
    m = J(os.path.join(RUN_G, "metrics_sentinel_groupsplit.json")); base = J(os.path.join(RUN_O, "metrics_sentinel.json"))["tta"]
    m1g = J(os.path.join(M1_G, "metrics_sentinel_groupsplit.json"))["tta"]
    shutil.copy(os.path.join(RUN_G, "metrics_sentinel_groupsplit.json"), os.path.join(out, "metrics.json"))
    shutil.copy(os.path.join(RUN_G, "per_tile_sentinel_groupsplit.csv"), os.path.join(out, "per_tile.csv"))
    shutil.copy(os.path.join(RUN_G, "train_log.csv"), os.path.join(out, "train_log.csv"))
    shutil.copy(os.path.join(RUN_G, "qualitative_sentinel_groupsplit.png"), os.path.join(out, "qualitative.png"))
    ep, loss, sec = training_curve(os.path.join(RUN_G, "train_log.csv"), os.path.join(out, "training_curve.png"),
                                   "M2: training loss, leak-free (group-disjoint) split")
    med, bad = per_tile_hist(os.path.join(RUN_G, "per_tile_sentinel_groupsplit.csv"), os.path.join(out, "per_tile_iou.png"))
    t = m["tta"]
    L = ["# M2 Improved DeepLabV3+: group-disjoint (leak-free) split, SOS Sentinel-1", "",
         "Same model, recipe and seed as the official run; only the split changes (the split used for M1: "
         "`leakage_check/splits/sos_sentinel_group_s0.csv`, train = partitions train+val+calib, "
         f"3,358 tiles). Test: {m['n_tiles']} tiles from sea areas never seen in training. {ep} epochs, {sec // 60} min.", "",
         "| Metric (%, with TTA) | M2 official split (leaky) | **M2 group-disjoint** | Difference | M1 group-disjoint |",
         "|---|---|---|---|---|"]
    for lab, k in [("**Oil IoU**", "iou_oil"), ("**Oil F1**", "f1_oil"), ("Precision (oil)", "precision_oil"),
                   ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("IoU (background)", "iou_bg"), ("Accuracy", "accuracy"),
                   ("Boundary F1 (2 px)", "boundary_f1"), ("ECE (×100)", "ece"), ("Oil share of test pixels", "oil_pixel_fraction")]:
        L.append(f"| {lab} | {pc(base[k])} | **{pc(t[k])}** | {dpp(base[k], t[k])} | {pc(m1g[k])} |")
    L += ["", "## Interpretation",
          f"- **The leak inflates M2's oil scores slightly** (oil IoU {dpp(base['iou_oil'], t['iou_oil'])} pp, F1 "
          f"{dpp(base['f1_oil'], t['f1_oil'])} pp): the same small effect as for M1. Recall falls, precision rises.",
          "- mIoU rises because the leak-free test set has less oil; oil IoU and F1 are the fair comparison.",
          f"- On the leak-free split M2 is slightly ahead of M1 (oil IoU {dpp(m1g['iou_oil'], t['iou_oil'])} pp, "
          f"F1 {dpp(m1g['f1_oil'], t['f1_oil'])} pp) with 5× fewer parameters. Single seed for both.", "",
          f"Figures: `training_curve.png`, `per_tile_iou.png` (median {med:.2f}; {100 * bad:.0f} % of oil tiles below 0.3), `qualitative.png`.", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(L))
    return dict(epochs=ep, loss=loss, sec=sec, med=med, bad=bad)


def peru():
    out = os.path.join(RES, "peru_external"); os.makedirs(out, exist_ok=True)
    M, S = {}, {}
    for tag, run, sos in [("official", RUN_O, "metrics_sentinel.json"), ("groupsplit", RUN_G, "metrics_sentinel_groupsplit.json")]:
        M[tag] = J(os.path.join(run, "peru", "metrics_peru.json")); S[tag] = J(os.path.join(run, sos))["tta"]
        shutil.copy(os.path.join(run, "peru", "metrics_peru.json"), os.path.join(out, f"metrics_{tag}.json"))
        shutil.copy(os.path.join(run, "peru", "qualitative_peru.png"), os.path.join(out, f"qualitative_{tag}.png"))
    m1 = {t: J(os.path.join(EXP, "M1_CBDNet", "results", "peru_external", f"metrics_{t}.json"))["all"] for t in M}
    o, g = M["official"]["all"], M["groupsplit"]["all"]
    L = ["# M2 Improved DeepLabV3+: external test on Peru (new sea)", "",
         "Same protocol as M1: all 2,013 Peru patches, oil vs everything else, 8-way TTA, frozen models.", "",
         "| Metric (%) | M2 official: SOS | **M2 official: Peru** | M2 leak-free: SOS | **M2 leak-free: Peru** | M1 official: Peru | M1 leak-free: Peru |",
         "|---|---|---|---|---|---|---|"]
    for lab, k in [("**Oil F1**", "f1_oil"), ("**Oil IoU**", "iou_oil"), ("Precision (oil)", "precision_oil"),
                   ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("ROC-AUC (×100)", "roc_auc"), ("ECE (×100)", "ece")]:
        L.append(f"| {lab} | {p1(S['official'][k])} | **{p1(o[k])}** | {p1(S['groupsplit'][k])} | **{p1(g[k])}** | "
                 f"{p1(m1['official'][k])} | {p1(m1['groupsplit'][k])} |")
    L += ["", "| True class | Called oil: M2 official | Called oil: M2 leak-free | M1 official | M1 leak-free |", "|---|---|---|---|---|"]
    for c in ["oil", "look-alike", "ship", "sea", "land"]:
        L.append(f"| {c} | {p1(o['called_oil_rate_per_class'][c])} | {p1(g['called_oil_rate_per_class'][c])} | "
                 f"{p1(m1['official']['called_oil_rate_per_class'][c])} | {p1(m1['groupsplit']['called_oil_rate_per_class'][c])} |")
    L += ["", "## Findings",
          f"1. **M2 transfers as badly as M1**: oil F1 {p1(o['f1_oil'])} / {p1(g['f1_oil'])} % (M1: "
          f"{p1(m1['official']['f1_oil'])} / {p1(m1['groupsplit']['f1_oil'])} %). Recall stays about 95 %, precision about 13 %.",
          f"2. **Look-alikes are called oil** ({p1(o['called_oil_rate_per_class']['look-alike'])}–"
          f"{p1(g['called_oil_rate_per_class']['look-alike'])} %), and the probabilities are over-confident (ECE about 0.30).",
          "3. Unlike M1, the leak-free M2 is **not** better on Peru. The M1 gain (+5 pp F1) may therefore be seed noise "
          "rather than an effect of the split.",
          "4. The failure is shared by two different architectures, so it comes from the training data (SOS has no "
          "look-alike class), not from one model design.", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(L))
    return M


def mdod():
    out = os.path.join(RES, "mdod_external"); os.makedirs(out, exist_ok=True)
    M = {}
    for tag, run in [("official", RUN_O), ("groupsplit", RUN_G)]:
        M[tag] = J(os.path.join(run, "mdod", "metrics_mdod.json"))
        shutil.copy(os.path.join(run, "mdod", "metrics_mdod.json"), os.path.join(out, f"metrics_{tag}.json"))
        shutil.copy(os.path.join(run, "mdod", "qualitative_mdod.png"), os.path.join(out, f"qualitative_{tag}.png"))
    m1 = {t: J(os.path.join(EXP, "M1_CBDNet", "results", "mdod_external", f"metrics_{t}.json")) for t in M}
    alert_chart(M, os.path.join(out, "alert_rates.png"))
    L = ["# M2 Improved DeepLabV3+: external test on MDOD (look-alike false alerts)", "",
         "Same protocol as M1: 400 boxes per class, single pass, a box alerts if at least 2 % of it has p(oil) > 0.5.", "",
         "![](alert_rates.png)", "",
         "| Box class | Alert @2 %: M2 official | Alert @2 %: M2 leak-free | M1 official | M1 leak-free |", "|---|---|---|---|---|"]
    for c in CLASSES:
        L.append(f"| {c} | {p1(M['official']['per_class'][c]['alert_rate@0.02'])} | {p1(M['groupsplit']['per_class'][c]['alert_rate@0.02'])} | "
                 f"{p1(m1['official']['per_class'][c]['alert_rate@0.02'])} | {p1(m1['groupsplit']['per_class'][c]['alert_rate@0.02'])} |")
    L += ["", "| AUC oil vs | M2 official | M2 leak-free | M1 official | M1 leak-free |", "|---|---|---|---|---|"]
    for c in LA + ["all look-alikes"]:
        L.append(f"| {c} | {M['official']['auc_oil_vs'][c]:.3f} | {M['groupsplit']['auc_oil_vs'][c]:.3f} | "
                 f"{m1['official']['auc_oil_vs'][c]:.3f} | {m1['groupsplit']['auc_oil_vs'][c]:.3f} |")
    L += ["", "## Findings",
          "1. **Same failure as M1:** every look-alike class is flagged more often than real oil, and AUC is below 0.5.",
          "2. M2 flags even fewer oil boxes than M1 (51–57 % vs 56–64 %); MDOD oil is often thin streaks (scale caveat as for M1).", ""]
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write("\n".join(L))
    return M


if __name__ == "__main__":
    print(sos_official(), sos_group()); peru(); mdod(); print("wrote", RES)
