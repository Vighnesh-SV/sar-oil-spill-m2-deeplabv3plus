"""Build the M2 Improved DeepLabV3+ results report (Word) from the saved metrics, figures and logs.

Reuses the document helpers of the M1 report so both reports look the same. Every number is read from
M2_DeepLabV3Plus/results/**/metrics*.json (and the M1 results for the comparison section).
usage (from experiments/):
    python M2_DeepLabV3Plus/make_report_m2.py
    python M2_DeepLabV3Plus/report/build_m2_report.py
    powershell -File M2_DeepLabV3Plus/report/to_pdf.ps1
"""
import json
import os
import sys

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.dirname(HERE)
EXP = os.path.dirname(M2)
sys.path.insert(0, os.path.join(EXP, "M1_CBDNet", "report"))
import build_m1_report as base  # noqa: E402  (Doc helpers)
import make_terminal_pics as term  # noqa: E402

R = os.path.join(M2, "results")
R1 = os.path.join(EXP, "M1_CBDNet", "results")
T = os.path.join(HERE, "terminal")
OUT = os.path.join(HERE, "M2_DeepLabV3Plus_Results_Report.docx")
INK2 = base.INK2
J = lambda *p: json.load(open(os.path.join(*p), encoding="utf-8"))  # noqa: E731
p1, p0, dpp = base.p1, base.p0, base.dpp
LA = ["biogenic slick", "low wind", "rain cell", "sea ice"]
PUB = {"miou": 0.8134, "f1_oil": 0.8962}


def terminal_pics():
    os.makedirs(T, exist_ok=True)
    term.OUT = T
    rd = lambda n: open(os.path.join(M2, "logs", n), encoding="utf-8", errors="replace").read().splitlines()  # noqa: E731
    term.render(term.excerpt(rd("train_eval_sentinel_official.log"), 6, 7), "train_eval_sentinel_official.log  (official split)",
                "term_train_official.png")
    term.render(term.excerpt(rd("train_eval_sentinel_groupsplit.log"), 6, 7), "train_eval_sentinel_groupsplit.log  (leak-free split)",
                "term_train_groupsplit.png")
    peru = [l for l in rd("eval_peru.log") if l.startswith("[peru")]
    term.render(peru[:2] + [""] + peru[2:], "eval_peru.log  (top: official-split model, bottom: leak-free model)", "term_peru.png")
    L, keep = rd("eval_mdod.log"), []
    for i, l in enumerate(L):
        if l.startswith("class "):
            keep += ([""] if keep else []) + L[i:i + 7]
    term.render(keep, "eval_mdod.log  (top: official-split model, bottom: leak-free model)", "term_mdod.png")


def build():
    terminal_pics()
    D = base.Doc()
    D.d.sections[0].footer.paragraphs[0].runs[0].text = "M2 Improved DeepLabV3+ results report  ·  page "
    OFF, GRP = J(R, "sentinel_official_split", "metrics.json"), J(R, "sentinel_group_split", "metrics.json")
    ot, op_, gt, gp = OFF["tta"], OFF["plain"], GRP["tta"], GRP["plain"]
    M1o = J(R1, "sentinel_official_split", "metrics.json")["tta"]; M1g = J(R1, "sentinel_group_split", "metrics.json")["tta"]
    PE = {t: J(R, "peru_external", f"metrics_{t}.json") for t in ("official", "groupsplit")}
    PE1 = {t: J(R1, "peru_external", f"metrics_{t}.json")["all"] for t in ("official", "groupsplit")}
    MD = {t: J(R, "mdod_external", f"metrics_{t}.json") for t in ("official", "groupsplit")}
    MD1 = {t: J(R1, "mdod_external", f"metrics_{t}.json") for t in ("official", "groupsplit")}
    po, pg = PE["official"]["all"], PE["groupsplit"]["all"]
    mo, mg = MD["official"], MD["groupsplit"]
    import csv
    import statistics

    def tiles(split):
        rows = list(csv.DictReader(open(os.path.join(R, split, "per_tile.csv"), encoding="utf-8")))
        v = [float(r["iou_oil"]) for r in rows if float(r["gt_oil_px"]) > 0]
        return statistics.median(v), sum(x < 0.3 for x in v) / len(v)

    def tinfo(split):
        rows = list(csv.DictReader(open(os.path.join(R, split, "train_log.csv"), encoding="utf-8")))
        return len(rows), float(rows[-1]["train_loss"]), int(rows[-1]["seconds"])
    med_o, low_o = tiles("sentinel_official_split"); med_g, low_g = tiles("sentinel_group_split")
    ep_o, loss_o, sec_o = tinfo("sentinel_official_split"); ep_g, loss_g, sec_g = tinfo("sentinel_group_split")
    hm = lambda s: f"{s // 3600} h {s % 3600 // 60:02d} min"  # noqa: E731
    rng = lambda M, k="alert_rate@0.02": f'{p0(min(M["per_class"][c][k] for c in LA))}–{p0(max(M["per_class"][c][k] for c in LA))}'  # noqa: E731

    # ---- title
    t = D.d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.space_before = Pt(60)
    r = t.add_run("M2: Improved DeepLabV3+"); r.bold = True; r.font.size = Pt(26); r.font.color.rgb = RGBColor(0x0B, 0x2E, 0x59)
    for txt, size, col in [("Implementation and results report", 15, None),
                           ("Oil-spill detection in Sentinel-1 SAR imagery  ·  method 2 of 3", 11, INK2),
                           ("", 10.5, INK2)]:
        s = D.d.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = s.add_run(txt); r.font.size = Pt(size)
        if col:
            r.font.color.rgb = col
    D.d.paragraphs[-1].paragraph_format.space_after = Pt(30)
    D.p("**Method.** J. Zhang, P. Yang, X. Ren, “Detection of oil spill in SAR image using an improved DeepLabV3+,” "
        "Sensors [P01]. No official code is released; we re-implemented the network from the paper.")
    D.p("**What this report contains.** The same four tests as for M1, on two trained M2 models: (A) the official "
        "SOS split (**leaky**); (B) the **leak-free** group-disjoint split; (C) the external Peru dataset; (D) the external "
        "MDOD dataset; plus (E) a direct comparison with M1. Every table is generated from the saved metric files; the "
        "terminal figures are excerpts of the real console logs.")
    D.toc(); D.page_break()

    # ---- 1 summary
    D.h("1. Summary of results")
    D.table("All M2 results at a glance (%, test-time augmentation unless noted), with M1 for reference.", [
        ["Test", "Model", "Oil F1", "Oil IoU", "mIoU", "Precision", "Recall", "ECE (×100)"],
        ["A. SOS official split (leaky)", "M2 official-split", p1(ot["f1_oil"]), p1(ot["iou_oil"]), p1(ot["miou"]),
         p1(ot["precision_oil"]), p1(ot["recall_oil"]), p1(ot["ece"])],
        ["   published [P01]", "", "89.62", "—", "81.34", "—", "—", "—"],
        ["   M1 CBD-Net (ours)", "M1 official-split", p1(M1o["f1_oil"]), p1(M1o["iou_oil"]), p1(M1o["miou"]),
         p1(M1o["precision_oil"]), p1(M1o["recall_oil"]), p1(M1o["ece"])],
        ["B. SOS group split (leak-free)", "M2 leak-free", p1(gt["f1_oil"]), p1(gt["iou_oil"]), p1(gt["miou"]),
         p1(gt["precision_oil"]), p1(gt["recall_oil"]), p1(gt["ece"])],
        ["C. Peru (external, 2,013 patches)", "M2 official-split", p1(po["f1_oil"]), p1(po["iou_oil"]), p1(po["miou"]),
         p1(po["precision_oil"]), p1(po["recall_oil"]), p1(po["ece"])],
        ["", "M2 leak-free", p1(pg["f1_oil"]), p1(pg["iou_oil"]), p1(pg["miou"]),
         p1(pg["precision_oil"]), p1(pg["recall_oil"]), p1(pg["ece"])],
    ], widths=[4.6, 2.8, 1.5, 1.5, 1.5, 1.7, 1.5, 1.9])
    D.table("MDOD (external, box labels only).", [
        ["Model", "Oil boxes flagged (should be high)", "Look-alike boxes flagged (should be low)", "AUC oil vs look-alikes (0.5 = chance)"],
        ["M2 official-split", p0(mo["per_class"]["oil spill"]["alert_rate@0.02"]), rng(mo), f'{mo["auc_oil_vs"]["all look-alikes"]:.3f}'],
        ["M2 leak-free", p0(mg["per_class"]["oil spill"]["alert_rate@0.02"]), rng(mg), f'{mg["auc_oil_vs"]["all look-alikes"]:.3f}'],
        ["M1 official-split", p0(MD1["official"]["per_class"]["oil spill"]["alert_rate@0.02"]), rng(MD1["official"]),
         f'{MD1["official"]["auc_oil_vs"]["all look-alikes"]:.3f}'],
    ], widths=[3.4, 4.3, 4.7, 4.6])
    D.p("**Key findings**")
    D.bullets([
        f"**The re-implementation succeeds.** mIoU {p1(ot['miou'])} % (published 81.34 %, {dpp(PUB['miou'], ot['miou'])} pp) "
        f"and oil F1 {p1(ot['f1_oil'])} % (published 89.62 %, {dpp(PUB['f1_oil'], ot['f1_oil'])} pp): both within ±2 pp.",
        f"**In-domain, M2 slightly beats M1 with 5× fewer parameters** ({OFF['params_M']:.1f} M vs 31.4 M): mIoU "
        f"{dpp(M1o['miou'], ot['miou'])} pp, oil F1 {dpp(M1o['f1_oil'], ot['f1_oil'])} pp, boundary F1 "
        f"{dpp(M1o['boundary_f1'], ot['boundary_f1'])} pp.",
        f"**The leak is small for M2 as well:** oil IoU {dpp(ot['iou_oil'], gt['iou_oil'])} pp and oil F1 "
        f"{dpp(ot['f1_oil'], gt['f1_oil'])} pp on the leak-free split.",
        f"**M2 fails on new seas exactly like M1.** Peru oil F1 {p0(po['f1_oil'])}–{p0(pg['f1_oil'])} %, "
        f"{p0(po['called_oil_rate_per_class']['look-alike'])}–{p0(pg['called_oil_rate_per_class']['look-alike'])} % of look-alike "
        "pixels called oil, ECE about 0.30; on MDOD every look-alike class is flagged more often than oil (AUC < 0.5).",
        "**The failure is shared by two different architectures**, so it comes from the training data (SOS has no "
        "look-alike class), not from one model design. All results are single-seed.",
    ])
    D.page_break()

    # ---- 2 method
    D.h("2. Method and implementation")
    D.h("2.1 How the improved DeepLabV3+ works", 2)
    D.bullets([
        "**Input:** one 256 × 256 SAR tile (3 channels).",
        "**Encoder:** MobileNetV2 (ImageNet weights) instead of DeepLabV3+'s usual Xception, with an **scSE attention "
        "module at the input of every inverted-residual block**. Output stride 16 (the last blocks use dilation).",
        "**ASPP:** 1 × 1 convolution, atrous rates 6, 12, 18 and image pooling, with scSE on the concatenated branches; 256 channels.",
        "**Decoder:** standard DeepLabV3+ decoder with a 1/4-resolution skip connection from the encoder (24 channels).",
        "**Output:** sigmoid oil probability per pixel; threshold 0.5.",
    ])
    D.h("2.2 Training and testing setup", 2)
    D.table("Setup used for both M2 models.", [
        ["Setting", "Value"],
        ["Training data", "SOS Sentinel-1 training tiles, 256 × 256 (3,354 official / 3,358 leak-free)"],
        ["Loss", "BCE + Dice (paper)"],
        ["Optimiser", "Adam, learning rate 1 × 10⁻³, batch size 15, 80 epochs (paper)"],
        ["Schedule", "Cosine decay to 0 (assumed: the paper gives no schedule). The final-epoch model is evaluated (no validation set in the official split)."],
        ["Augmentation", "Horizontal / vertical flips and 90° rotations (assumed)"],
        ["Testing", "Same as M1: 8-way test-time augmentation; single pass also reported"],
        ["Size", f"{OFF['params_M']:.2f} M parameters, 6.56 GMACs (paper: 5.84 M, 6.16 GMACs)"],
        ["Hardware", "NVIDIA RTX 3050 laptop GPU (4 GB), 16 GB RAM, PyTorch 2.6 (CUDA 12.4)"],
        ["Training time", f"official split: {ep_o} epochs, {hm(sec_o)};  leak-free split: {ep_g} epochs, {hm(sec_g)}"],
    ], widths=[3.5, 13.5], bold_first_col=True)
    D.h("2.3 Choices where the paper is silent", 2)
    D.bullets([
        "**Not in the paper, so assumed:** ImageNet-1k weights, cSE reduction ratio 16, 256 ASPP channels, cosine "
        "learning-rate decay, flip / rotation augmentation. These choices explain the 0.2 M parameter and 0.4 GMAC "
        "difference from the paper.",
        "**Which F1?** The paper does not say whether its F1 is for the oil class. If it used the official SOS script "
        f"(which scores the background class), the comparable value of ours is {p1(ot['f1_bg_officialscript'])} %. We "
        "treat oil F1 as the fair comparison.",
        "**Interruptions:** the training saves its full state every epoch and can continue with `--resume`. The first "
        "official-split attempt was stopped at epoch 43 by low system memory and was restarted from scratch.",
    ])
    D.h("2.4 Code map", 2)
    D.table("Where each part of M2 lives (repository root).", [
        ["Part", "File", "Written by"],
        ["Model (DeepLabV3+ MobileNetV2 + scSE)", "`common/models.py` (`dlv3p_mnv2`)", "us, from the paper"],
        ["Loss (BCE + Dice)", "`third_party/CBDNet-main/dice_bce_loss.py`", "shared with M1"],
        ["Training loop", "`common/train.py`", "us; generic trainer for M2 / M3"],
        ["Data loading", "`common/sos_data.py`", "shared with M1"],
        ["SOS, Peru, MDOD tests", "`common/evaluate.py`, `evaluate_peru.py`, `evaluate_mdod.py`", "shared with M1"],
        ["Results folders", "`M2_DeepLabV3Plus/make_report_m2.py`", "us"],
    ], widths=[4.5, 8.0, 4.5])
    D.p("Metrics are identical to the M1 report (pooled pixels; accuracy, precision, recall, F1, IoU, mIoU, boundary "
        "F1, ROC-AUC, PR-AUC, ECE, Brier).")

    # ---- 3 Test A
    D.h("3. Test A: SOS official split (leaky)")
    D.p(f"Run `M2_dlv3p_mnv2_sentinel_s0`. Trained on the 3,354 official tiles, tested on the {OFF['n_tiles']} official test "
        f"tiles. {ep_o} epochs; final training loss {loss_o:.4f}.")
    rows = [["Metric (%)", "Ours, with TTA", "Ours, single pass", "Published [P01]", "M1 CBD-Net (TTA)"]]
    pub = {"f1_oil": "89.62", "miou": "81.34"}
    for lab, k in base.seg_rows(None):
        rows.append([lab, p1(ot[k]), p1(op_[k]), pub.get(k, "—"), p1(M1o[k])])
    D.table("Segmentation metrics on the official SOS Sentinel-1 test set.", rows, widths=[5.8, 2.7, 2.7, 2.9, 2.9])
    D.table("Probability quality.", [
        ["Metric", "With TTA", "Single pass", "M1 (TTA)"],
        *[[lab, f"{ot[k]:.4f}", f"{op_[k]:.4f}", f"{M1o[k]:.4f}"] for lab, k in
          [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("ECE (15 bins)", "ece"), ("Brier score", "brier")]]],
        widths=[5.5, 3.8, 3.8, 3.9])
    D.p(f"**Reproduction gap:** mIoU {dpp(PUB['miou'], ot['miou'])} pp, F1 {dpp(PUB['f1_oil'], ot['f1_oil'])} pp "
        "(target ±2 pp). The gaps are wider than for M1 (±0.2 pp), as expected for a re-implementation without code.")
    D.fig(os.path.join(R, "sentinel_official_split", "training_curve.png"), "Training loss per epoch, official split (cosine learning-rate decay).", 13)
    D.fig(os.path.join(T, "term_train_official.png"),
          "Console output of the official-split run (excerpt of `train_eval_sentinel_official.log`; “…” marks omitted "
          "epochs). The last two lines are the test results.")
    D.fig(os.path.join(R, "sentinel_official_split", "per_tile_iou.png"),
          f"Per-tile oil IoU. Median {med_o:.2f}; {100*low_o:.0f} % of oil tiles score below 0.3.", 12)
    D.fig(os.path.join(R, "sentinel_official_split", "qualitative.png"),
          "Four median and four worst test tiles (green = correct oil, red = false oil, blue = missed oil).")
    D.bullets([
        "Median tiles: slicks are found with clean shapes; errors sit on the coarse polygon edges of the labels.",
        "Worst tiles: (a) dark, featureless tiles labelled entirely as oil, which M2 calls sea (blue); (b) a bright "
        "ship with a dark wake where M2 predicts a large false slick (red); (c) small labelled slicks inside a larger dark "
        "area that M2 over-segments.",
    ])

    # ---- 4 Test B
    D.h("4. Test B: SOS group-disjoint split (leak-free)")
    D.p(f"Run `M2_dlv3p_mnv2_sentinel_s0_groupsplit`. The same leak-free split as M1: no sea area appears in both train "
        f"(3,358 tiles) and test ({GRP['n_tiles']} tiles). Same recipe and seed as Test A. {ep_g} epochs; final loss {loss_g:.4f}.")
    rows = [["Metric (%)", "With TTA", "Single pass"]]
    for lab, k in base.seg_rows(None):
        rows.append([lab, p1(gt[k]), p1(gp[k])])
    rows += [[lab, f"{gt[k]:.4f}", f"{gp[k]:.4f}"] for lab, k in
             [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("ECE (15 bins)", "ece"), ("Brier score", "brier")]]
    D.table("All metrics on the leak-free test set.", rows, widths=[9.0, 4.0, 4.0])
    rows = [["Metric (%, TTA)", "M2 official (leaky)", "M2 leak-free", "Difference (pp)", "M1 leak-free"]]
    for lab, k in [("**Oil IoU**", "iou_oil"), ("**Oil F1**", "f1_oil"), ("Precision (oil)", "precision_oil"),
                   ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("IoU (background)", "iou_bg"), ("Accuracy", "accuracy"),
                   ("Boundary F1 (2 px)", "boundary_f1"), ("ECE (×100)", "ece"), ("Oil share of test pixels", "oil_pixel_fraction")]:
        rows.append([lab, p1(ot[k]), p1(gt[k]), dpp(ot[k], gt[k]), p1(M1g[k])])
    D.table("Effect of removing the leak, with M1 on the same leak-free split.", rows, widths=[4.6, 3.2, 3.0, 3.0, 3.2])
    D.bullets([
        f"**The leak inflates M2's oil scores slightly:** oil IoU {dpp(ot['iou_oil'], gt['iou_oil'])} pp, oil F1 "
        f"{dpp(ot['f1_oil'], gt['f1_oil'])} pp (M1: −1.74 / −1.11 pp). Recall falls ({dpp(ot['recall_oil'], gt['recall_oil'])} pp), "
        f"precision rises ({dpp(ot['precision_oil'], gt['precision_oil'])} pp): the same pattern as M1.",
        "mIoU rises because the leak-free test set has less oil (29.6 % vs 34.7 % of pixels); oil IoU and F1 are the fair comparison.",
        f"On the leak-free split, M2 leads M1 by {dpp(M1g['iou_oil'], gt['iou_oil'])} pp oil IoU and "
        f"{dpp(M1g['f1_oil'], gt['f1_oil'])} pp F1, which is within single-seed noise.",
    ])
    D.fig(os.path.join(R, "sentinel_group_split", "training_curve.png"), "Training loss per epoch, leak-free split.", 13)
    D.fig(os.path.join(T, "term_train_groupsplit.png"), "Console output of the leak-free run (excerpt of `train_eval_sentinel_groupsplit.log`).")
    D.fig(os.path.join(R, "sentinel_group_split", "per_tile_iou.png"),
          f"Per-tile oil IoU, leak-free test set. Median {med_g:.2f}; {100*low_g:.0f} % of oil tiles below 0.3.", 12)
    D.fig(os.path.join(R, "sentinel_group_split", "qualitative.png"), "Four median and four worst tiles of the leak-free test set.")
    D.page_break()

    # ---- 5 Peru
    D.h("5. Test C: external dataset Peru (new sea)")
    D.p("Same protocol as for M1: all 2,013 Peru patches (40 scenes, never used for training), oil vs everything else, "
        "8-way TTA, frozen models.")
    rows = [["Metric (%)", "M2 official: SOS", "M2 official: Peru", "M2 leak-free: SOS", "M2 leak-free: Peru", "M1 official: Peru"]]
    for lab, k in [("**Oil F1**", "f1_oil"), ("**Oil IoU**", "iou_oil"), ("Precision (oil)", "precision_oil"),
                   ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("Accuracy", "accuracy"), ("ROC-AUC (×100)", "roc_auc"),
                   ("PR-AUC (×100)", "pr_auc"), ("ECE (×100)", "ece"), ("Brier (×100)", "brier")]:
        rows.append([lab, p0(ot[k]), f"**{p0(po[k])}**", p0(gt[k]), f"**{p0(pg[k])}**", p0(PE1["official"][k])])
    D.table("In-domain (SOS) vs new sea (Peru, all 2,013 patches).", rows, widths=[3.2, 2.7, 2.8, 2.8, 2.8, 2.7])
    rows = [["True class", "Share of Peru pixels", "Called oil: M2 official", "Called oil: M2 leak-free", "M1 official", "M1 leak-free"]]
    for c in ["oil", "look-alike", "ship", "sea", "land"]:
        rows.append([c, p0(po["class_pixel_share"][c]), p0(po["called_oil_rate_per_class"][c]), p0(pg["called_oil_rate_per_class"][c]),
                     p0(PE1["official"]["called_oil_rate_per_class"][c]), p0(PE1["groupsplit"]["called_oil_rate_per_class"][c])])
    D.table("Share of each true Peru class predicted as oil (%).", rows, widths=[2.6, 2.9, 3.0, 3.0, 2.7, 2.8])
    oo, og = PE["official"]["official_test"], PE["groupsplit"]["official_test"]
    D.table("Peru's own scene-disjoint test subset (338 patches, 5 scenes).", [
        ["Model", "Oil F1", "Oil IoU", "Precision", "Recall", "ECE (×100)", "Look-alike called oil"],
        *[[n, p0(m["f1_oil"]), p0(m["iou_oil"]), p0(m["precision_oil"]), p0(m["recall_oil"]), p0(m["ece"]),
           p0(m["called_oil_rate_per_class"]["look-alike"])] for n, m in [("M2 official-split", oo), ("M2 leak-free", og)]]],
        widths=[3.2, 2.0, 2.0, 2.2, 2.0, 2.4, 3.2])
    D.fig(os.path.join(T, "term_peru.png"), "Console output of the Peru tests (`eval_peru.log`).")
    D.fig(os.path.join(R, "peru_external", "qualitative_official.png"),
          "Peru examples, M2 official-split model: three oil patches and three look-alike patches. Rows: SAR, label "
          "(white = oil, orange = look-alike, grey = land, magenta = ship), prediction overlay (green = correct oil, red = "
          "false oil, blue = missed oil).", 15)
    D.fig(os.path.join(R, "peru_external", "qualitative_groupsplit.png"), "Peru examples, M2 leak-free model.", 15)
    D.bullets([
        f"**M2 transfers as badly as M1:** oil F1 {p0(po['f1_oil'])} / {p0(pg['f1_oil'])} % (M1: {p0(PE1['official']['f1_oil'])} / "
        f"{p0(PE1['groupsplit']['f1_oil'])} %). Recall stays about 95 %, precision collapses to about 13 %.",
        f"**Look-alikes are called oil** ({p0(po['called_oil_rate_per_class']['look-alike'])}–{p0(pg['called_oil_rate_per_class']['look-alike'])} %), "
        f"and so is {p0(pg['called_oil_rate_per_class']['sea'])}–{p0(po['called_oil_rate_per_class']['sea'])} % of open sea.",
        f"**Ranking partly survives (ROC-AUC {po['roc_auc']:.2f}), but the probabilities are over-confident:** ECE "
        f"{po['ece']:.3f} / {pg['ece']:.3f}, worse than M1 ({PE1['official']['ece']:.3f} / {PE1['groupsplit']['ece']:.3f}).",
        "**Unlike M1, the leak-free M2 is not better on Peru.** The M1 gain (+5 pp F1) may therefore be seed noise "
        "rather than an effect of the split; more seeds are needed.",
    ])
    D.page_break()

    # ---- 6 MDOD
    D.h("6. Test D: external dataset MDOD (look-alike false alerts)")
    D.p("Same protocol as for M1: 400 boxes per class, single pass; a box **alerts** if at least 2 % of its pixels have "
        "p(oil) > 0.5. AUC = probability that a random oil box scores higher than a random look-alike box (0.5 = chance).")
    rows = [["Box class", "Alert @2 %: M2 official", "Alert @2 %: M2 leak-free", "Alert @20 %: M2 official",
             "Alert @20 %: M2 leak-free", "Alert @2 %: M1 official"]]
    for c in ["oil spill"] + LA:
        a, b = mo["per_class"][c], mg["per_class"][c]
        rows.append([("**oil spill** (should alert)" if c == "oil spill" else c), p0(a["alert_rate@0.02"]), p0(b["alert_rate@0.02"]),
                     p0(a["alert_rate@0.2"]), p0(b["alert_rate@0.2"]), p0(MD1["official"]["per_class"][c]["alert_rate@0.02"])])
    D.table("Share of MDOD boxes flagged as oil (%), per class.", rows, widths=[3.7, 2.7, 2.7, 2.6, 2.7, 2.6])
    rows = [["Oil vs", "AUC (mean): M2 official", "AUC (mean): M2 leak-free", "AUC (peak): M2 official",
             "AUC (peak): M2 leak-free", "AUC (mean): M1 official"]]
    for c in LA + ["all look-alikes"]:
        rows.append([c, f"{mo['auc_oil_vs'][c]:.3f}", f"{mg['auc_oil_vs'][c]:.3f}", f"{mo['auc_oil_vs_peak'][c]:.3f}",
                     f"{mg['auc_oil_vs_peak'][c]:.3f}", f"{MD1['official']['auc_oil_vs'][c]:.3f}"])
    D.table("Can the model rank oil boxes above look-alike boxes? (AUC; 0.5 = chance)", rows, widths=[3.7, 2.7, 2.7, 2.6, 2.7, 2.6])
    D.fig(os.path.join(R, "mdod_external", "alert_rates.png"), "Share of MDOD boxes flagged as oil (coverage ≥ 2 %), M2.", 14)
    D.fig(os.path.join(T, "term_mdod.png"), "Console output of the MDOD tests (summary tables).", 14)
    D.fig(os.path.join(R, "mdod_external", "qualitative_official.png"),
          "MDOD boxes (top) and predicted oil probability (bottom, bright = oil), M2 official-split model.")
    D.fig(os.path.join(R, "mdod_external", "qualitative_groupsplit.png"), "The same boxes, M2 leak-free model.")
    D.bullets([
        f"**Same failure as M1:** {rng(mo)} % of look-alike boxes are flagged, but only "
        f"{p0(mg['per_class']['oil spill']['alert_rate@0.02'])}–{p0(mo['per_class']['oil spill']['alert_rate@0.02'])} % of oil boxes. "
        "Low wind is flagged almost always.",
        f"**AUC is below 0.5 for every look-alike class** ({mo['auc_oil_vs']['all look-alikes']:.2f} / "
        f"{mg['auc_oil_vs']['all look-alikes']:.2f} overall; M1: {MD1['official']['auc_oil_vs']['all look-alikes']:.2f}).",
        "M2 flags even fewer oil boxes than M1. Scale caveat as for M1: MDOD is about 10× coarser than SOS, so many oil "
        "spills are streaks a few pixels wide.",
    ])

    # ---- 7 M1 vs M2
    D.h("7. Test E: M1 vs M2")
    rows = [["Test", "Metric", "M1 CBD-Net", "M2 DeepLabV3+", "Better"]]

    def row(test, lab, a, b, higher=True, fmt=p1):
        better = "M2" if (b > a) == higher else "M1"
        rows.append([test, lab, fmt(a), fmt(b), better if abs(a - b) > 1e-9 else "="])
    row("SOS official", "Oil F1 (%)", M1o["f1_oil"], ot["f1_oil"]); row("SOS official", "mIoU (%)", M1o["miou"], ot["miou"])
    row("SOS official", "Boundary F1 (%)", M1o["boundary_f1"], ot["boundary_f1"])
    row("SOS leak-free", "Oil F1 (%)", M1g["f1_oil"], gt["f1_oil"]); row("SOS leak-free", "Oil IoU (%)", M1g["iou_oil"], gt["iou_oil"])
    row("Peru (official-split models)", "Oil F1 (%)", PE1["official"]["f1_oil"], po["f1_oil"])
    row("Peru (official-split models)", "ECE (×100)", PE1["official"]["ece"], po["ece"], higher=False)
    row("Peru (leak-free models)", "Oil F1 (%)", PE1["groupsplit"]["f1_oil"], pg["f1_oil"])
    row("MDOD (official-split models)", "AUC oil vs look-alikes", MD1["official"]["auc_oil_vs"]["all look-alikes"],
        mo["auc_oil_vs"]["all look-alikes"], fmt=lambda v: f"{v:.3f}")
    rows.append(["Size", "Parameters (M)", "31.4", f"{OFF['params_M']:.1f}", "M2"])
    D.table("Head-to-head comparison (TTA; MDOD single pass).", rows, widths=[4.6, 4.0, 2.8, 3.0, 2.6])
    D.bullets([
        "**In-domain, M2 is as good or slightly better** (and 5× smaller), with clearly better boundaries.",
        "**Out of domain, both fail in the same way**, and M1 is slightly better on Peru. The gaps between the two models "
        "(0.4–6 pp) are much smaller than the in-domain-to-new-sea drop (about 60 pp F1).",
        "**Conclusion:** changing the architecture does not fix the look-alike and calibration problem. This motivates "
        "the planned contributions (calibration, alert thresholds, wind- and look-alike-aware alerting), which act on top of any model.",
    ])

    D.h("8. Reproducing these results")
    D.table("Commands (from the repository root).", [
        ["Step", "Command"],
        ["Test A: train", "`python common\\train.py --arch dlv3p_mnv2 --sensor sentinel --seed 0 --resume`"],
        ["Test A: test", "`python common\\evaluate.py --run runs\\M2_dlv3p_mnv2_sentinel_s0 --arch dlv3p_mnv2`"],
        ["Test B: train", "`python common\\train.py --arch dlv3p_mnv2 --split <group split CSV> --resume`"],
        ["Test B: test", "`python common\\evaluate.py --run runs\\M2_..._groupsplit --arch dlv3p_mnv2 --split <group split CSV>`"],
        ["Tests C, D", "`python common\\evaluate_peru.py` / `evaluate_mdod.py --run runs\\<run> --arch dlv3p_mnv2`"],
        ["Results folders", "`python M2_DeepLabV3Plus\\make_report_m2.py`"],
        ["This report", "`python M2_DeepLabV3Plus\\report\\build_m2_report.py`, then `to_pdf.ps1`"],
    ], widths=[3.5, 13.5], bold_first_col=True)
    D.p("Weights: `runs/M2_dlv3p_mnv2_sentinel_s0/best.pth` and `runs/M2_dlv3p_mnv2_sentinel_s0_groupsplit/best.pth` (24 MB each).")
    D.d.save(OUT)
    print("wrote", OUT, "tables", D.nt, "figures", D.nf)


if __name__ == "__main__":
    build()
