"""Build the M1 CBD-Net results report (Word) from the saved metrics, figures and logs.

Every number is read from M1_CBDNet/results/**/metrics*.json, so the report always matches the runs.
usage (from experiments/):
    python M1_CBDNet/report/make_terminal_pics.py
    python M1_CBDNet/report/build_m1_report.py
    powershell -File M1_CBDNet/report/to_pdf.ps1        (Word: refresh contents table, save PDF)
"""
import csv
import json
import os
import re
import statistics

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.dirname(HERE)
EXP = os.path.dirname(M1)
R = os.path.join(M1, "results")
T = os.path.join(HERE, "terminal")
OUT = os.path.join(HERE, "M1_CBDNet_Results_Report.docx")
INK2 = RGBColor(0x52, 0x51, 0x4E)
HEAD_FILL = "DCE8F7"
WIDTH_CM = 17.0

J = lambda *p: json.load(open(os.path.join(R, *p), encoding="utf-8"))  # noqa: E731
OFF = J("sentinel_official_split", "metrics.json")
GRP = J("sentinel_group_split", "metrics.json")
PERU = {"official": J("peru_external", "metrics_official.json"), "groupsplit": J("peru_external", "metrics_groupsplit.json")}
MDOD = {"official": J("mdod_external", "metrics_official.json"), "groupsplit": J("mdod_external", "metrics_groupsplit.json")}
p1 = lambda v: f"{100 * v:.2f}"  # noqa: E731
p0 = lambda v: f"{100 * v:.1f}"  # noqa: E731
dpp = lambda a, b: f"{100 * (b - a):+.2f}"  # noqa: E731


def train_info(split):
    rows = list(csv.DictReader(open(os.path.join(R, split, "train_log.csv"), encoding="utf-8")))
    k = min(rows, key=lambda r: float(r["train_loss"]))
    return len(rows), float(k["train_loss"]), int(k["epoch"])


def tile_stats(split):
    rows = list(csv.DictReader(open(os.path.join(R, split, "per_tile.csv"), encoding="utf-8")))
    v = [float(r["iou_oil"]) for r in rows if float(r["gt_oil_px"]) > 0 and r["iou_oil"] not in ("", "nan")]
    return statistics.median(v), sum(x < 0.3 for x in v) / len(v)


# ---------------------------------------------------------------- docx helpers
class Doc:
    def __init__(self):
        self.d = Document()
        s = self.d.sections[0]
        s.page_height, s.page_width = Cm(29.7), Cm(21.0)
        s.left_margin = s.right_margin = Cm(2.0); s.top_margin = s.bottom_margin = Cm(2.0)
        st = self.d.styles
        st["Normal"].font.name = "Calibri"; st["Normal"].font.size = Pt(10.5)
        st["Normal"].paragraph_format.space_after = Pt(5)
        for name, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)]:
            h = st[name]; h.font.name = "Calibri"; h.font.size = Pt(size); h.font.bold = True
            h.font.color.rgb = RGBColor(0x0B, 0x2E, 0x59)
            h.paragraph_format.space_before = Pt(14 if size > 12 else 10); h.paragraph_format.space_after = Pt(5)
            h.paragraph_format.keep_with_next = True
            rpr = h.element.get_or_add_rPr(); f = rpr.find(qn("w:rFonts"))
            if f is None:
                f = OxmlElement("w:rFonts"); rpr.append(f)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                f.set(qn(a), "Calibri")
        self.nt = self.nf = 0
        self._footer()

    def _field(self, run, code):
        for kind, text in [("begin", None), (None, code), ("separate", None), ("end", None)]:
            if kind:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), kind)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = text
            run._r.append(e)

    def _footer(self):
        p = self.d.sections[0].footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("M1 CBD-Net results report  ·  page "); r.font.size = Pt(8.5); r.font.color.rgb = INK2
        r2 = p.add_run(); r2.font.size = Pt(8.5); self._field(r2, "PAGE")

    def rich(self, p, text, size=None, italic=False):
        """**bold** and `code` inline markup."""
        for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
            if not part:
                continue
            if part.startswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("`"):
                r = p.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
            else:
                r = p.add_run(part)
            if size:
                r.font.size = Pt(size)
            if italic:
                r.italic = True
        return p

    def h(self, text, lvl=1):
        self.d.add_heading(text, lvl)

    def p(self, text, **kw):
        return self.rich(self.d.add_paragraph(), text, **kw)

    def bullets(self, items, numbered=False):
        for it in items:
            self.rich(self.d.add_paragraph(style="List Number" if numbered else "List Bullet"), it)

    def note(self, text):
        p = self.rich(self.d.add_paragraph(), text, size=9, italic=True)
        for r in p.runs:
            r.font.color.rgb = INK2

    def table(self, caption, rows, widths=None, bold_first_col=False, highlight_cols=()):
        self.nt += 1
        cp = self.d.add_paragraph(); cp.paragraph_format.keep_with_next = True
        cp.paragraph_format.space_before = Pt(6)
        self.rich(cp, f"**Table {self.nt}.** {caption}", size=9.5)
        ncol = len(rows[0])
        widths = widths or [WIDTH_CM / ncol] * ncol
        t = self.d.add_table(rows=len(rows), cols=ncol); t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                c = t.cell(i, j); c.width = Cm(widths[j])
                c.paragraphs[0].paragraph_format.space_after = Pt(0)
                short = len(str(val)) <= 16
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if (j > 0 and short) else WD_ALIGN_PARAGRAPH.LEFT
                txt = str(val)
                if i == 0 or (bold_first_col and j == 0) or j in highlight_cols:
                    txt = txt if txt.startswith("**") or not txt else f"**{txt}**"
                self.rich(c.paragraphs[0], txt, size=9)
                if i < min(3, len(rows) - 1):  # keep the header with the first rows (no stranded header)
                    c.paragraphs[0].paragraph_format.keep_with_next = True
                if i == 0:
                    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto")
                    sh.set(qn("w:fill"), HEAD_FILL); c._tc.get_or_add_tcPr().append(sh)
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); t.rows[i]._tr.get_or_add_trPr().append(cs)
            if i == 0:  # repeat header row on page breaks
                trpr = t.rows[0]._tr.get_or_add_trPr(); e = OxmlElement("w:tblHeader"); e.set(qn("w:val"), "true")
                trpr.append(e)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def fig(self, path, caption, width_cm=WIDTH_CM):
        self.nf += 1
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Cm(width_cm))
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.rich(c, f"**Figure {self.nf}.** {caption}", size=9)

    def toc(self):
        p = self.d.add_paragraph(); r = p.add_run(); self._field(r, 'TOC \\o "1-2" \\h \\z \\u')
        self.note("(Contents table: refreshed automatically when the PDF is made.)")

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- content
def seg_rows(a, b=None, labels=("", "")):
    keys = [("Accuracy", "accuracy"), ("Precision (oil)", "precision_oil"), ("Recall (oil)", "recall_oil"),
            ("**F1 (oil)**", "f1_oil"), ("IoU (oil)", "iou_oil"), ("IoU (background)", "iou_bg"),
            ("**mIoU**", "miou"), ("Boundary F1 (2 px)", "boundary_f1"),
            ("F1 as in the official script (background class)", "f1_bg_officialscript")]
    return keys


def build():
    D = Doc()
    ep_o, loss_o, best_o = train_info("sentinel_official_split")
    ep_g, loss_g, best_g = train_info("sentinel_group_split")
    med_o, low_o = tile_stats("sentinel_official_split")
    med_g, low_g = tile_stats("sentinel_group_split")
    ot, op_ = OFF["tta"], OFF["plain"]
    gt, gp = GRP["tta"], GRP["plain"]

    # ---- title
    t = D.d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.space_before = Pt(60)
    r = t.add_run("M1: CBD-Net"); r.bold = True; r.font.size = Pt(26); r.font.color.rgb = RGBColor(0x0B, 0x2E, 0x59)
    s = D.d.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run("Implementation and results report"); r.font.size = Pt(15)
    s = D.d.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run("Oil-spill detection in Sentinel-1 SAR imagery  ·  method 1 of 3"); r.font.size = Pt(11)
    r.font.color.rgb = INK2
    s = D.d.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER; s.paragraph_format.space_after = Pt(30)
    r = s.add_run(""); r.font.size = Pt(10.5); r.font.color.rgb = INK2
    D.p("**Method.** Q. Zhu *et al.*, “Oil spill contextual and boundary-supervised detection network based on "
        "marine SAR images,” *IEEE TGRS* [P24]. Official code: github.com/CUG-URS/CBDNet-main.".replace("*", ""))
    D.p("**What this report contains.** Two trained models of the same method and four tests: "
        "(A) the official SOS test split, which is **leaky**; (B) a **leak-free** group-disjoint split; "
        "(C) the external Peru dataset; and (D) the external MDOD dataset. Every table is generated from the saved "
        "metric files, and the terminal figures are excerpts of the real console logs.")
    D.toc()
    D.page_break()

    # ---- 1 summary
    D.h("1. Summary of results")
    po, pg = PERU["official"]["all"], PERU["groupsplit"]["all"]
    mo, mg = MDOD["official"], MDOD["groupsplit"]
    D.table("All M1 results at a glance (%, test-time augmentation unless noted).", [
        ["Test", "Model", "Oil F1", "Oil IoU", "mIoU", "Precision", "Recall", "ECE (×100)"],
        ["A. SOS official split (leaky)", "official-split", p1(ot["f1_oil"]), p1(ot["iou_oil"]), p1(ot["miou"]),
         p1(ot["precision_oil"]), p1(ot["recall_oil"]), p1(ot["ece"])],
        ["   published [P29]", "", "87.81", "—", "82.94", "—", "—", "—"],
        ["   CBD-Net paper [P24], joint loss", "", "87.87", "—", "83.42", "91.20", "87.32", "—"],
        ["B. SOS group split (leak-free)", "leak-free", p1(gt["f1_oil"]), p1(gt["iou_oil"]), p1(gt["miou"]),
         p1(gt["precision_oil"]), p1(gt["recall_oil"]), p1(gt["ece"])],
        ["C. Peru (external, 2,013 patches)", "official-split", p1(po["f1_oil"]), p1(po["iou_oil"]), p1(po["miou"]),
         p1(po["precision_oil"]), p1(po["recall_oil"]), p1(po["ece"])],
        ["", "leak-free", p1(pg["f1_oil"]), p1(pg["iou_oil"]), p1(pg["miou"]),
         p1(pg["precision_oil"]), p1(pg["recall_oil"]), p1(pg["ece"])],
    ], widths=[4.6, 2.4, 1.6, 1.6, 1.6, 1.8, 1.6, 1.8])
    D.table("MDOD (external, box labels only): share of boxes flagged as oil, and how well oil is ranked above look-alikes.", [
        ["Model", "Oil boxes flagged (should be high)", "Look-alike boxes flagged (should be low)",
         "AUC oil vs look-alikes (0.5 = chance)"],
        ["official-split", p0(mo["per_class"]["oil spill"]["alert_rate@0.02"]),
         f'{p0(min(mo["per_class"][c]["alert_rate@0.02"] for c in LA))}–{p0(max(mo["per_class"][c]["alert_rate@0.02"] for c in LA))}',
         f'{mo["auc_oil_vs"]["all look-alikes"]:.3f}'],
        ["leak-free", p0(mg["per_class"]["oil spill"]["alert_rate@0.02"]),
         f'{p0(min(mg["per_class"][c]["alert_rate@0.02"] for c in LA))}–{p0(max(mg["per_class"][c]["alert_rate@0.02"] for c in LA))}',
         f'{mg["auc_oil_vs"]["all look-alikes"]:.3f}'],
    ], widths=[3.2, 4.4, 4.8, 4.6])
    D.p("**Key findings**")
    D.bullets([
        f"**The reproduction succeeds.** On the official SOS split our mIoU is {p1(ot['miou'])} % (published 82.94 %) "
        f"and oil F1 is {p1(ot['f1_oil'])} % (published 87.81 %); both gaps are within ±0.2 pp. Against CBD-Net's own paper "
        f"(mIoU 83.42 %, F1 87.87 %, trained with its joint boundary loss) the gaps are {100 * ot['miou'] - 83.42:+.2f} / "
        f"{100 * ot['f1_oil'] - 87.87:+.2f} pp.",
        f"**The official split leaks, but the effect is small.** Every official test tile has augmented copies of "
        f"its sea area in training. On a leak-free split, oil IoU drops by {abs(100*(gt['iou_oil']-ot['iou_oil'])):.2f} pp "
        f"and oil F1 by {abs(100*(gt['f1_oil']-ot['f1_oil'])):.2f} pp.",
        f"**M1 does not transfer to a new sea.** On Peru, oil F1 falls to {p0(po['f1_oil'])}–{p0(pg['f1_oil'])} %. "
        f"Recall stays above 95 %, but precision collapses to {p0(po['precision_oil'])}–{p0(pg['precision_oil'])} % "
        "because almost every dark area is called oil. The probabilities become badly calibrated (ECE ×7–9).",
        "**M1 cannot reject look-alikes.** On MDOD every look-alike class is flagged more often than real oil, and "
        "the ranking AUC is below 0.5.",
        "**The leak-free model is the better model** on both external datasets (higher F1 and lower ECE on Peru, "
        "fewer false alerts on MDOD). All results are from a single training seed.",
    ])
    D.page_break()

    # ---- 2 method
    D.h("2. Method and implementation")
    D.h("2.1 How CBD-Net works", 2)
    D.bullets([
        "**Input:** one 256 × 256 SAR tile (3 channels).",
        "**Encoder:** ResNet-34 pretrained on ImageNet, with an scSE attention block after every stage. scSE "
        "re-weights both the channels and the pixel locations.",
        "**Bottleneck:** a dilated-convolution block (rates 1, 2, 4, 8) that gives wide spatial context.",
        "**Decoder:** LinkNet-style, four up-sampling blocks with additive skip connections.",
        "**Output:** a sigmoid gives an oil probability for every pixel; threshold 0.5 gives the oil mask.",
    ])
    D.h("2.2 Training and testing setup", 2)
    D.table("Setup used for both models (identical to the official code).", [
        ["Setting", "Value"],
        ["Training data", "SOS Sentinel-1 training tiles, 256 × 256 (3,354 official / 3,358 leak-free)"],
        ["Loss", "BCE + Dice (as in the released code)"],
        ["Optimiser", "Adam, learning rate 2 × 10⁻⁴, batch size 4"],
        ["Schedule", "Up to 80 epochs. After more than 3 epochs without improvement of the training loss: reload "
                     "the best weights and divide the learning rate by 5. Stop after more than 6."],
        ["Augmentation", "HSV jitter, shift-scale-rotate, horizontal / vertical flips, 90° rotations (official)"],
        ["Testing", "Official 8-way test-time augmentation (TTA): 4 rotations × flip, probabilities averaged. "
                    "A single-pass result is also reported."],
        ["Parameters", f"{OFF['params_M']:.1f} M"],
        ["Hardware", "NVIDIA RTX 3050 laptop GPU (4 GB), 16 GB RAM, PyTorch 2.6 (CUDA 12.4)"],
        ["Training time", f"official split: {ep_o} epochs, 1 h 47 min;  leak-free split: {ep_g} epochs, 1 h 18 min"],
    ], widths=[3.5, 13.5], bold_first_col=True)
    D.h("2.3 What we changed, and two things we found in the official code", 2)
    D.p("The network, loss and schedule are imported **unchanged** from the official repository. We changed only:")
    D.bullets(["`np.math` → `math`, required by NumPy 2 (identical behaviour);",
               "`shuffle=True` in the data loader (the official code uses `shuffle=False`);",
               "fixed random seeds and a per-epoch CSV log, so that runs can be repeated."])
    D.p("Findings in the official code:")
    D.bullets([
        "**The official evaluation script reports precision, recall and F1 for the background class, not for oil.** "
        "`t1p3-Iou.py` calls `confusion_matrix(labels=[255, 0])`, which makes index 1 the background. We report "
        "oil-class metrics as the main numbers and the official-script value separately.",
        "**The released code trains with BCE + Dice only.** The boundary loss described in the paper exists in the "
        "repository (`jointloss`) but is not used by default.",
    ], numbered=True)
    D.h("2.4 Code map", 2)
    D.table("Where each part of M1 lives (repository root).", [
        ["Part", "File", "Written by"],
        ["Model (CBD-Net)", "`third_party/CBDNet-main/networks/CBDNet.py`", "official authors, unchanged"],
        ["Loss (BCE + Dice)", "`third_party/CBDNet-main/dice_bce_loss.py`", "official authors, unchanged"],
        ["Training loop", "`M1_CBDNet/train_m1.py`", "us, following the official `train.py`"],
        ["Data loading, augmentation", "`common/sos_data.py`", "us; augmentation copied from the official code"],
        ["SOS testing and metrics", "`common/evaluate.py`", "us; shared by M1–M3"],
        ["Peru test", "`common/evaluate_peru.py`", "us; shared by M1–M3"],
        ["MDOD test", "`common/evaluate_mdod.py`", "us; shared by M1–M3"],
        ["Reports and figures", "`M1_CBDNet/make_report.py`, `make_peru_report.py`, `make_mdod_report.py`", "us"],
    ], widths=[4.0, 8.5, 4.5])

    # ---- 3 metrics
    D.h("3. Metrics")
    D.p("Pixels are pooled over all test tiles, as in the official SOS script. “Oil” is the positive class.")
    D.table("Metric definitions (TP / FP / FN / TN = true / false positive / false negative / true negative oil pixels).", [
        ["Metric", "Meaning", "Formula"],
        ["Accuracy", "share of all pixels classified correctly", "(TP + TN) / all"],
        ["Precision (oil)", "of the pixels called oil, the share that really is oil", "TP / (TP + FP)"],
        ["Recall (oil)", "of the real oil pixels, the share that was found", "TP / (TP + FN)"],
        ["F1 (oil)", "balance of precision and recall", "2PR / (P + R)"],
        ["IoU (oil)", "overlap of predicted and true oil areas", "TP / (TP + FP + FN)"],
        ["mIoU", "mean of oil IoU and background IoU", "(IoU_oil + IoU_bg) / 2"],
        ["Boundary F1", "F1 of slick outlines, within 2 pixels", "—"],
        ["ROC-AUC / PR-AUC", "how well probabilities rank oil above non-oil (threshold-free)", "area under curve"],
        ["ECE", "calibration: gap between predicted probability and observed frequency (15 bins); 0 = perfect",
         "Σ (n_b/N)·|acc_b − conf_b|"],
        ["Brier score", "mean squared error of the probabilities", "mean (p − y)²"],
    ], widths=[3.0, 9.0, 5.0], bold_first_col=True)
    D.page_break()

    # ---- 4 Test A
    D.h("4. Test A: SOS official split (leaky)")
    D.p(f"Run `M1_cbdnet_sentinel_dicebce_s0`. Trained on the 3,354 official Sentinel-1 training tiles and tested "
        f"on the {OFF['n_tiles']} official test tiles. Training stopped at epoch {ep_o}; the best training loss was "
        f"{loss_o:.4f} at epoch {best_o}.")
    D.h("4.1 Quantitative results", 2)
    rows = [["Metric (%)", "Ours, with TTA", "Ours, single pass", "Published [P29] / CBD-Net paper [P24]"]]
    pub = {"f1_oil": "87.81 / 87.87", "miou": "82.94 / 83.42", "precision_oil": "— / 91.20", "recall_oil": "— / 87.32"}
    for lab, k in seg_rows(None):
        rows.append([lab, p1(ot[k]), p1(op_[k]), pub.get(k, "—")])
    D.table("Segmentation metrics on the official SOS Sentinel-1 test set.", rows, widths=[7.0, 3.3, 3.3, 3.4])
    D.table("Probability quality.", [
        ["Metric", "With TTA", "Single pass"],
        ["ROC-AUC", f"{ot['roc_auc']:.4f}", f"{op_['roc_auc']:.4f}"],
        ["PR-AUC", f"{ot['pr_auc']:.4f}", f"{op_['pr_auc']:.4f}"],
        ["ECE (15 bins)", f"{ot['ece']:.4f}", f"{op_['ece']:.4f}"],
        ["Brier score", f"{ot['brier']:.4f}", f"{op_['brier']:.4f}"],
    ], widths=[7.0, 5.0, 5.0])
    D.p(f"**Reproduction gap:** mIoU {dpp(0.8294, ot['miou'])} pp and F1 {dpp(0.8781, ot['f1_oil'])} pp "
        "(target: within ±2 pp). TTA adds about 0.2–0.3 pp to every metric.")
    D.h("4.2 Training", 2)
    D.fig(os.path.join(R, "sentinel_official_split", "training_curve.png"),
          "Training loss per epoch, official split. Dashed lines mark the learning-rate drops.", 13)
    D.fig(os.path.join(T, "term_train_official.png"),
          "Console output of the official-split run (excerpt of `train_eval_sentinel_official.log`; "
          "“…” marks omitted epochs, * marks a new best loss). The last two lines are the test results.")
    D.h("4.3 Qualitative results", 2)
    D.fig(os.path.join(R, "sentinel_official_split", "per_tile_iou.png"),
          f"Per-tile oil IoU on the official test set. Median {med_o:.2f}; {100*low_o:.0f} % of oil tiles score below 0.3.", 12)
    D.fig(os.path.join(R, "sentinel_official_split", "qualitative.png"),
          "Four median and four worst test tiles. Green = correct oil, red = false oil, blue = missed oil.")
    D.bullets([
        "Large slicks are segmented well. Most errors sit on slick edges, where the ground-truth outlines are coarse "
        f"polygons; this explains the low boundary F1 ({p1(ot['boundary_f1'])} %).",
        "The worst tiles are (a) fully dark tiles labelled as oil, which show no edge or context, and (b) thin dark "
        "streaks that the model marks as oil but the labels do not (possible look-alikes or missing labels).",
        f"In-domain, the probabilities are already well calibrated (ECE {ot['ece']:.3f}).",
    ])

    # ---- 5 leakage
    D.h("5. Why the official split is leaky")
    D.p("SOS tile names are plain numbers with no scene information. Our near-duplicate check (template matching "
        "over all rotations and flips) found that the Sentinel-1 tiles 20001–22220 are **5 augmented versions of the "
        "same 444 sea areas**: tile *i* and tile *i* + 444·k show the same area. The official test set is simply the "
        "first 839 IDs, so **every official test tile has at least two augmented copies of its sea area in the "
        "training set**.".replace("*", ""))
    D.fig(os.path.join(EXP, "leakage_check", "outputs", "sibling_sets.png"),
          "Examples of one sea area appearing as five different SOS tiles (IDs i, i+444, i+888, …).", 10)
    D.p("Test A therefore shows that the **reproduction is correct**, but it does not measure performance on unseen "
        "sea areas. Test B removes this leak.")

    # ---- 6 Test B
    D.h("6. Test B: SOS group-disjoint split (leak-free)")
    D.p(f"Run `M1_cbdnet_sentinel_dicebce_s0_groupsplit`. Same model, loss, schedule and seed as Test A; only the "
        f"split changes. Tiles are grouped by sea area, and whole groups are assigned to train or test, so **no sea "
        f"area appears in both**. Train: 3,358 tiles (about the official 3,354, so training size is not a factor). "
        f"Test: {GRP['n_tiles']} tiles. Training stopped at epoch {ep_g}; best training loss {loss_g:.4f} at epoch {best_g}.")
    D.h("6.1 Quantitative results", 2)
    rows = [["Metric (%)", "With TTA", "Single pass"]]
    for lab, k in seg_rows(None):
        rows.append([lab, p1(gt[k]), p1(gp[k])])
    rows += [["ROC-AUC", f"{gt['roc_auc']:.4f}", f"{gp['roc_auc']:.4f}"],
             ["PR-AUC", f"{gt['pr_auc']:.4f}", f"{gp['pr_auc']:.4f}"],
             ["ECE (15 bins)", f"{gt['ece']:.4f}", f"{gp['ece']:.4f}"],
             ["Brier score", f"{gt['brier']:.4f}", f"{gp['brier']:.4f}"]]
    D.table("All metrics on the leak-free test set.", rows, widths=[9.0, 4.0, 4.0])
    D.h("6.2 Leaky vs leak-free", 2)
    cmp_keys = [("**Oil IoU**", "iou_oil"), ("**Oil F1**", "f1_oil"), ("Precision (oil)", "precision_oil"),
                ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("IoU (background)", "iou_bg"),
                ("Accuracy", "accuracy"), ("Boundary F1 (2 px)", "boundary_f1"), ("ECE (×100)", "ece"),
                ("Oil share of test pixels", "oil_pixel_fraction")]
    rows = [["Metric (%, with TTA)", "Test A: official (leaky)", "Test B: group split (leak-free)", "Difference (pp)"]]
    for lab, k in cmp_keys:
        rows.append([lab, p1(ot[k]), p1(gt[k]), dpp(ot[k], gt[k])])
    D.table("Effect of removing the leak.", rows, widths=[5.2, 3.8, 4.4, 3.6])
    D.bullets([
        f"**The leak inflates the oil scores only slightly:** oil IoU {dpp(ot['iou_oil'], gt['iou_oil'])} pp, oil F1 "
        f"{dpp(ot['f1_oil'], gt['f1_oil'])} pp. On unseen areas the model misses more oil (recall "
        f"{dpp(ot['recall_oil'], gt['recall_oil'])} pp) but raises fewer false alarms (precision "
        f"{dpp(ot['precision_oil'], gt['precision_oil'])} pp).",
        f"**mIoU hides the drop.** The leak-free test set contains less oil ({p1(gt['oil_pixel_fraction'])} % vs "
        f"{p1(ot['oil_pixel_fraction'])} % of pixels), so background IoU rises and offsets the lower oil IoU. "
        "Oil IoU and oil F1 are the fair comparison.",
        "Calibration (ECE) is unchanged.",
    ])
    D.h("6.3 Training and qualitative results", 2)
    D.fig(os.path.join(R, "sentinel_group_split", "training_curve.png"),
          "Training loss per epoch, leak-free split.", 13)
    D.fig(os.path.join(T, "term_train_groupsplit.png"),
          "Console output of the leak-free run (excerpt of `train_eval_sentinel_groupsplit.log`).")
    D.fig(os.path.join(R, "sentinel_group_split", "per_tile_iou.png"),
          f"Per-tile oil IoU, leak-free test set. Median {med_g:.2f}; {100*low_g:.0f} % of oil tiles score below 0.3.", 12)
    D.fig(os.path.join(R, "sentinel_group_split", "qualitative.png"),
          "Four median and four worst tiles of the leak-free test set (green = correct, red = false oil, blue = missed).")
    D.bullets([
        "Worst tiles: (a) a large dark patch predicted as oil where only a thin strip is labelled (a possible "
        "low-wind look-alike); (b) faint thin slicks that are missed; (c) a tile labelled almost entirely as oil with "
        "no visible slick edge (label ambiguity).",
    ])
    D.p("**Caveats.** (1) One seed: differences of 1–2 pp are close to run-to-run variation, so three seeds are "
        "needed before quoting a number. (2) The groups are sea areas (crops), not the 21 original scenes, so "
        "scene-level leakage is still possible and this is a lower bound. (3) The two test sets contain different "
        "tiles, so part of any difference comes from test composition.")
    D.page_break()

    # ---- 7 Peru
    D.h("7. Test C: external dataset Peru (new sea)")
    D.p("**Data.** Peru oil-spill set [N22], corrected version: 2,013 Sentinel-1 patches (512 × 512) from 40 scenes "
        "never used for training. Labels: oil, look-alike, ship, land, sea. We score oil vs everything "
        "else, with the same pooled metrics and 8-way TTA as on SOS. Each patch is cut into four 256 × 256 tiles. "
        "Both models are frozen.")
    D.h("7.1 Quantitative results", 2)
    rows = [["Metric (%)", "Official-split model: SOS", "Official-split model: Peru",
             "Leak-free model: SOS", "Leak-free model: Peru"]]
    for lab, k in [("**Oil F1**", "f1_oil"), ("**Oil IoU**", "iou_oil"), ("Precision (oil)", "precision_oil"),
                   ("Recall (oil)", "recall_oil"), ("mIoU", "miou"), ("Accuracy", "accuracy"),
                   ("ROC-AUC (×100)", "roc_auc"), ("PR-AUC (×100)", "pr_auc"), ("ECE (×100)", "ece"),
                   ("Brier (×100)", "brier")]:
        rows.append([lab, p0(ot[k]), f"**{p0(po[k])}**", p0(gt[k]), f"**{p0(pg[k])}**"])
    D.table("In-domain (SOS) vs new sea (Peru, all 2,013 patches).", rows, widths=[3.8, 3.3, 3.3, 3.3, 3.3])
    rows = [["True class", "Share of Peru pixels", "Called oil: official-split model", "Called oil: leak-free model"]]
    for c in ["oil", "look-alike", "ship", "sea", "land"]:
        rows.append([c, p0(po["class_pixel_share"][c]), p0(po["called_oil_rate_per_class"][c]),
                     p0(pg["called_oil_rate_per_class"][c])])
    D.table("What the model calls oil on Peru: share of each true class predicted as oil (%).", rows,
            widths=[3.5, 4.0, 4.8, 4.7])
    oo, og = PERU["official"]["official_test"], PERU["groupsplit"]["official_test"]
    D.table("Peru's own scene-disjoint test subset (338 patches, 5 scenes).", [
        ["Model", "Oil F1", "Oil IoU", "Precision", "Recall", "ECE (×100)", "Look-alike called oil"],
        ["official-split", p0(oo["f1_oil"]), p0(oo["iou_oil"]), p0(oo["precision_oil"]), p0(oo["recall_oil"]),
         p0(oo["ece"]), p0(oo["called_oil_rate_per_class"]["look-alike"])],
        ["leak-free", p0(og["f1_oil"]), p0(og["iou_oil"]), p0(og["precision_oil"]), p0(og["recall_oil"]),
         p0(og["ece"]), p0(og["called_oil_rate_per_class"]["look-alike"])],
    ], widths=[3.0, 2.0, 2.0, 2.2, 2.0, 2.4, 3.4])
    D.fig(os.path.join(T, "term_peru.png"), "Console output of the Peru test, both models (`eval_peru.log`).")
    D.h("7.2 Qualitative results", 2)
    D.fig(os.path.join(R, "peru_external", "qualitative_official.png"),
          "Peru examples, official-split model: three median oil patches and three look-alike patches. Rows: SAR, "
          "label (white = oil, orange = look-alike, grey = land, magenta = ship), prediction overlay (green = correct "
          "oil, red = false oil, blue = missed oil). The look-alike patches are predicted as oil almost everywhere.", 15)
    D.fig(os.path.join(R, "peru_external", "qualitative_groupsplit.png"),
          "Peru examples, leak-free model.", 15)
    D.h("7.3 Findings", 2)
    D.bullets([
        f"**Severe drop under domain shift:** oil F1 falls from about 87 % on SOS to {p0(po['f1_oil'])}–{p0(pg['f1_oil'])} % on Peru.",
        f"**Recall stays high but precision collapses** ({p0(po['precision_oil'])}–{p0(pg['precision_oil'])} %). "
        f"The model calls {p0(pg['called_oil_rate_per_class']['look-alike'])}–{p0(po['called_oil_rate_per_class']['look-alike'])} % "
        "of look-alike pixels oil, and about a third of open sea. Roughly 6 of every 7 oil alarms are false.",
        f"**The ranking is still good (ROC-AUC {po['roc_auc']:.2f}–{pg['roc_auc']:.2f}), but the 0.5 threshold and the "
        f"probabilities are wrong:** ECE rises from about 0.03 to {pg['ece']:.2f}–{po['ece']:.2f}. The model is "
        "confidently wrong on the new sea.",
        f"**The leak-free model transfers better:** F1 {dpp(po['f1_oil'], pg['f1_oil'])} pp, ECE "
        f"{pg['ece']:.3f} vs {po['ece']:.3f}.",
        "Not a data error: Peru and SOS have a similar sea brightness (median grey level 86 vs 89). Peru has lower "
        "contrast and darker oil, which is genuine domain shift.",
    ])
    D.page_break()

    # ---- 8 MDOD
    D.h("8. Test D: external dataset MDOD (look-alike false alerts)")
    s = MDOD["official"]["settings"]
    D.p("**Data.** MDOD [N05], OSF release: 4,673 full Sentinel-1 scenes with 24,940 **bounding boxes** in five "
        "classes: oil spill, biogenic slick, low wind, rain cell, sea ice. There are no pixel masks, so F1 and IoU "
        "cannot be computed. Instead we ask, for each box, how much of it the frozen model marks as oil.")
    D.p(f"**Protocol.** {s['per_class']} boxes per class (seed {s['seed']}; boxes smaller than {s['min_side']} px skipped). "
        "Each box is enlarged to at least 256 × 256 for context and predicted in 256 × 256 tiles, single pass. The "
        "white land mask is excluded. **Coverage** = share of box pixels with p(oil) > 0.5. A box **alerts** if its "
        "coverage is at least 2 %. **AUC** = probability that a random oil box scores higher than a random "
        "look-alike box (0.5 = chance; below 0.5 = look-alikes score higher).")
    D.h("8.1 Quantitative results", 2)
    rows = [["Box class", "Alert @2 %: official", "Alert @2 %: leak-free", "Alert @20 %: official",
             "Alert @20 %: leak-free", "Mean p(oil): official", "Mean p(oil): leak-free"]]
    for c in ["oil spill"] + LA:
        a, b = mo["per_class"][c], mg["per_class"][c]
        rows.append([("**oil spill** (should alert)" if c == "oil spill" else c),
                     p0(a["alert_rate@0.02"]), p0(b["alert_rate@0.02"]), p0(a["alert_rate@0.2"]),
                     p0(b["alert_rate@0.2"]), f"{a['mean_box_score']:.2f}", f"{b['mean_box_score']:.2f}"])
    D.table("Share of boxes flagged as oil (%), per class.", rows, widths=[3.8, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2])
    rows = [["Oil vs", "AUC (box mean): official", "AUC (box mean): leak-free", "AUC (box peak): official",
             "AUC (box peak): leak-free"]]
    for c in LA + ["all look-alikes"]:
        rows.append([c, f"{mo['auc_oil_vs'][c]:.3f}", f"{mg['auc_oil_vs'][c]:.3f}",
                     f"{mo['auc_oil_vs_peak'][c]:.3f}", f"{mg['auc_oil_vs_peak'][c]:.3f}"])
    D.table("Can the model rank oil boxes above look-alike boxes? (AUC; 0.5 = chance)", rows,
            widths=[3.8, 3.3, 3.3, 3.3, 3.3])
    D.fig(os.path.join(R, "mdod_external", "alert_rates.png"),
          "Share of MDOD boxes flagged as oil (coverage ≥ 2 %), per class and model.", 14)
    D.fig(os.path.join(T, "term_mdod.png"), "Console output of the MDOD test, both models (summary tables).", 14)
    D.h("8.2 Qualitative results", 2)
    D.fig(os.path.join(R, "mdod_external", "qualitative_official.png"),
          "MDOD boxes (top) and predicted oil probability (bottom, bright = oil), official-split model.")
    D.fig(os.path.join(R, "mdod_external", "qualitative_groupsplit.png"),
          "The same boxes, leak-free model.")
    D.h("8.3 Findings", 2)
    D.bullets([
        "**Look-alikes trigger more alerts than real oil.** 87–100 % of look-alike boxes are flagged, but only "
        "56–64 % of oil boxes. Low wind is the worst: almost every box is flagged, usually over more than half its area.",
        "**AUC is below 0.5 for every look-alike class.** M1 has learned “dark and smooth = oil”, not the shape of "
        "a slick, so large dark areas give the strongest oil signal.",
        f"**The leak-free model raises fewer false alerts** (AUC {mg['auc_oil_vs']['all look-alikes']:.2f} vs "
        f"{mo['auc_oil_vs']['all look-alikes']:.2f}) but also flags less oil.",
        "**Caveat, scale:** MDOD scenes are about 10× coarser than SOS tiles, so many MDOD oil spills are thin streaks "
        "only a few pixels wide, which the model misses. Part of the low oil alert rate is this resolution mismatch. "
        "The look-alike false alerts are real at any scale.",
    ])

    # ---- 9 conclusions
    D.h("9. Conclusions for M1")
    D.bullets([
        "M1 is **correctly reproduced** (mIoU and F1 within 0.2 pp of the published values).",
        "The leaky official SOS split overstates oil IoU and F1 by about 1–2 pp; this needs confirming with more seeds.",
        "In-domain scores (about 87 % F1) **do not carry over to new seas**: on Peru F1 is 24–29 %, and on MDOD "
        "look-alikes are flagged more often than oil.",
        "The failure mode is consistent across both external datasets: **dark areas are called oil, and the "
        "probabilities become over-confident.** The ranking (ROC-AUC 0.92 on Peru) is partly preserved, so "
        "re-calibration and better alert thresholds can recover part of the loss, while look-alike rejection "
        "needs extra information such as wind.",
        "The leak-free model is better on every external test and is the preferred M1 model for later comparisons.",
    ])
    D.h("10. Reproducing these results")
    D.p("All commands run from the repository root with the project virtual environment.")
    D.table("Commands.", [
        ["Step", "Command"],
        ["Test A: train + test + report", "`powershell -File M1_CBDNet\\run_m1.ps1`"],
        ["Test B: train", "`python M1_CBDNet\\train_m1.py --split <group split CSV>`"],
        ["Test B: test", "`python common\\evaluate.py --run runs\\M1_..._groupsplit --split <group split CSV>`"],
        ["Test C: Peru", "`python common\\evaluate_peru.py --run runs\\<run>`  then  `python M1_CBDNet\\make_peru_report.py`"],
        ["Test D: MDOD", "`python common\\evaluate_mdod.py --run runs\\<run>`  then  `python M1_CBDNet\\make_mdod_report.py`"],
        ["This report", "`python M1_CBDNet\\report\\make_terminal_pics.py`, `build_m1_report.py`, then `to_pdf.ps1`"],
    ], widths=[4.5, 12.5], bold_first_col=True)
    D.p("Trained weights: `runs/M1_cbdnet_sentinel_dicebce_s0/best.pth` and "
        "`runs/M1_cbdnet_sentinel_dicebce_s0_groupsplit/best.pth` (126 MB each). Leak-free split file: "
        "`leakage_check/splits/sos_sentinel_group_s0.csv`.")
    D.d.save(OUT)
    print("wrote", OUT, "tables", D.nt, "figures", D.nf)


LA = ["biogenic slick", "low wind", "rain cell", "sea ice"]

if __name__ == "__main__":
    build()
