"""Build ALL_RESULTS/: one Word file with every results table per method (M1, M2) + one analysis file.

Every table uses ONE fixed order so the documents can be compared line by line:
  models : (1) trained on the LEAKY official split, (2) trained on the LEAK-FREE group split
  tests  : SOS official test -> SOS leak-free test -> Peru (all 2,013) -> Peru test subset (338) -> MDOD
  metrics: Accuracy, Precision, Recall, F1, IoU oil, IoU background, mIoU, Boundary F1, F1 (official script),
           ROC-AUC, PR-AUC, ECE, MSE (= Brier), R²
All numbers are read from runs/**/metrics*.json, results/summary.csv and results/analysis/analysis.json.
usage (from experiments/):  python tools/analysis_overfit.py ; python tools/build_all_results.py
"""
import csv
import json
import os
import statistics
import sys

from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EXP, "M1_CBDNet", "report"))
sys.path.insert(0, os.path.join(EXP, "common"))
import build_m1_report as base  # noqa: E402

OUT = os.path.join(EXP, "docs"); os.makedirs(OUT, exist_ok=True)
RUNS = os.path.join(EXP, "runs")
AN = json.load(open(os.path.join(EXP, "results", "analysis", "analysis.json")))
SUMMARY = list(csv.DictReader(open(os.path.join(EXP, "results", "summary.csv"))))

METRICS = [("Accuracy", "accuracy"), ("Precision (oil)", "precision_oil"), ("Recall (oil)", "recall_oil"),
           ("F1 (oil)", "f1_oil"), ("IoU (oil)", "iou_oil"), ("IoU (background)", "iou_bg"), ("mIoU", "miou"),
           ("Boundary F1", "boundary_f1"), ("F1, official script (background class)", "f1_bg_officialscript"),
           ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("ECE", "ece"), ("MSE (= Brier score)", "brier"), ("R²", "r2")]
PCT = {"accuracy", "precision_oil", "recall_oil", "f1_oil", "iou_oil", "iou_bg", "miou", "boundary_f1", "f1_bg_officialscript"}
SHORT = ["Acc", "P", "R", "F1", "IoU oil", "IoU bg", "mIoU", "BF1", "F1 off.", "ROC-AUC", "PR-AUC", "ECE", "MSE", "R²"]
CLASSES = ["oil spill", "biogenic slick", "low wind", "rain cell", "sea ice"]
PERU_CLASSES = ["oil", "look-alike", "ship", "sea", "land"]
MODELS = [("leaky", "Trained on LEAKY official split"), ("leakfree", "Trained on LEAK-FREE group split")]

METHODS = {
    "M1": dict(title="M1: CBD-Net", paper="Q. Zhu et al., IEEE TGRS [P24]; official code, unchanged",
               runs={"leaky": "M1_cbdnet_sentinel_dicebce_s0", "leakfree": "M1_cbdnet_sentinel_dicebce_s0_groupsplit"},
               arch="cbdnet", published={"miou": 0.8294, "f1_oil": 0.8781}, pub_src="independent re-run of CBD-Net in [P29]",
               setup="Adam 2e-4, batch 4, up to 80 epochs, reduce-on-plateau of the training loss (÷5, reload best), "
                     "early stop; official augmentation; BCE + Dice; best-training-loss checkpoint",
               key="M1", file="M1_CBDNet_All_Results.docx"),
    "M2": dict(title="M2: Improved DeepLabV3+", paper="J. Zhang et al., Sensors [P01]; re-implemented (no code released)",
               runs={"leaky": "M2_dlv3p_mnv2_sentinel_s0", "leakfree": "M2_dlv3p_mnv2_sentinel_s0_groupsplit"},
               arch="dlv3p_mnv2", published={"miou": 0.8134, "f1_oil": 0.8962}, pub_src="Zhang et al. [P01]",
               setup="Adam 1e-3, batch 15, 80 epochs, cosine decay (assumed); flips + 90° rotations (assumed); BCE + Dice; "
                     "final-epoch checkpoint", key="M2", file="M2_DeepLabV3Plus_All_Results.docx"),
}


def fmt(k, v):
    if v is None or v == "":
        return "—"
    v = float(v)
    if k == "r2":
        return f"{v:.3f}"
    return f"{100 * v:.2f}" if k in PCT else f"{v:.4f}"


def summ(run, test, mode):
    for r in SUMMARY:
        if r["run"] == run and r["test_set"] == test and r["mode"] == mode:
            return r
    return None


def J(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


def r2_of(mse, f):
    """Pixel-level R² of the probabilities vs the 0/1 label: 1 − MSE / Var(label), Var = f(1 − f), f = oil share."""
    return 1 - float(mse) / (f * (1 - f))


def oil_frac(run, test):
    if test.startswith("sentinel"):
        return J(RUNS, run, f"metrics_{test}.json")["tta"]["oil_pixel_fraction"]
    if test.startswith("peru"):
        return J(RUNS, run, "peru", "metrics_peru.json")["all" if test == "peru_all" else "official_test"]["oil_pixel_fraction"]
    return None


for _r in SUMMARY:
    _f = oil_frac(_r["run"], _r["test_set"])
    _r["r2"] = r2_of(_r["brier"], _f) if _f else ""


def landscape_doc(title_txt, footer):
    D = base.Doc()
    s = D.d.sections[0]
    s.orientation = WD_ORIENT.LANDSCAPE; s.page_width, s.page_height = Cm(29.7), Cm(21.0)
    s.left_margin = s.right_margin = Cm(1.6); s.top_margin = s.bottom_margin = Cm(1.6)
    D.d.sections[0].footer.paragraphs[0].runs[0].text = footer + "  ·  page "
    t = D.d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(title_txt); r.bold = True; r.font.size = Pt(20); r.font.color.rgb = RGBColor(0x0B, 0x2E, 0x59)
    return D


W = 26.5  # usable width (cm) in landscape


def glossary(D):
    D.h("Metric definitions (same order as every table)", 2)
    D.table("Metrics. TP / FP / FN / TN = oil pixels predicted correctly / false oil / missed oil / correct background. "
            "Pixels are pooled over all test tiles. Higher is better except ECE and MSE.", [
                ["Metric", "Short", "What it measures", "Formula", "Good value"],
                ["Accuracy", "Acc", "share of all pixels classified correctly (inflated by the large sea background)", "(TP+TN)/all", "high, but not informative alone"],
                ["Precision (oil)", "P", "of the pixels called oil, how many are oil (false-alarm control)", "TP/(TP+FP)", "high"],
                ["Recall (oil)", "R", "of the real oil pixels, how many were found (missed-oil control)", "TP/(TP+FN)", "high"],
                ["F1 (oil)", "F1", "balance of precision and recall; equals the Dice coefficient", "2PR/(P+R)", "high"],
                ["IoU (oil)", "IoU oil", "overlap of predicted and true oil areas; stricter than F1", "TP/(TP+FP+FN)", "high"],
                ["IoU (background)", "IoU bg", "same for the background class", "TN/(TN+FP+FN)", "high"],
                ["mIoU", "mIoU", "mean of oil IoU and background IoU; the main SOS benchmark number. Rises when the test set has more background", "(IoU oil + IoU bg)/2", "high"],
                ["Boundary F1", "BF1", "F1 of the slick outlines, matched within 2 pixels", "—", "high"],
                ["F1, official script", "F1 off.", "what the official SOS script reports: F1 of the BACKGROUND class (a bug in that script)", "2·Pbg·Rbg/(Pbg+Rbg)", "for comparison only"],
                ["ROC-AUC", "ROC-AUC", "how well probabilities rank oil pixels above non-oil, at any threshold", "area under ROC", "1.0 (0.5 = chance)"],
                ["PR-AUC", "PR-AUC", "same, focused on the oil class; sensitive to class imbalance", "area under precision-recall", "high"],
                ["ECE", "ECE", "calibration: gap between predicted probability and how often it is right (15 bins)", "Σ (n_b/N)|acc_b − conf_b|", "0"],
                ["MSE (= Brier score)", "MSE", "mean squared error between predicted oil probability p and true label y (0/1) of every pixel", "mean (p − y)²", "0"],
                ["R²", "R²", "share of the label variance explained by the probabilities; 0 = no better than predicting the average oil share, negative = worse", "1 − MSE / (f(1 − f)), f = oil share", "1"],
                ["TTA", "", "test-time augmentation: average of 8 rotated / flipped predictions", "", ""],
                ["Alert rate (MDOD)", "", "share of boxes in which at least x % of pixels have p(oil) > 0.5", "", "high for oil, low for look-alikes"],
                ["AUC oil vs class (MDOD)", "", "chance that a random oil box scores higher than a random box of that class", "Mann-Whitney", "1.0 (0.5 = chance)"],
            ], widths=[3.4, 1.6, 11.5, 5.0, 5.0], bold_first_col=True)


def method_doc(M):
    D = landscape_doc(f"{M['title']}: all results tables", f"{M['title']}: all results")
    s = D.d.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run(f"{M['paper']}  ·  generated from the saved metric files"); r.font.color.rgb = base.INK2
    D.p("**Fixed order in every table.** Models: (1) trained on the **LEAKY** official SOS split, (2) trained on the "
        "**LEAK-FREE** group-disjoint split. Tests: SOS official test → SOS leak-free test → Peru all → Peru test subset "
        "→ MDOD. Metrics: Acc, P, R, F1, IoU oil, IoU bg, mIoU, BF1, F1 official script, ROC-AUC, PR-AUC, ECE, MSE, R² "
        "(percentages for the first nine). Peru and MDOD were never used for training.")
    glossary(D)
    runs = M["runs"]

    # ---------- T: master table
    D.h("1. Master table: every test, every metric", 1)
    rows = [["Test", "Model", "Mode"] + SHORT]
    plan = [("SOS official test (839 tiles)", "leaky", "sentinel", ("tta", "plain")),
            ("SOS leak-free test (835 tiles)", "leakfree", "sentinel_groupsplit", ("tta", "plain")),
            ("Peru, all 2,013 patches", "leaky", "peru_all", ("tta",)),
            ("Peru, all 2,013 patches", "leakfree", "peru_all", ("tta",)),
            ("Peru test subset (338 patches)", "leaky", "peru_official_test", ("tta",)),
            ("Peru test subset (338 patches)", "leakfree", "peru_official_test", ("tta",))]
    for test, mk, ts, modes in plan:
        for mode in modes:
            r = summ(runs[mk], ts, mode)
            rows.append([test, "leaky-trained" if mk == "leaky" else "leak-free-trained", mode.upper() if mode == "tta" else "single"]
                        + [fmt(k, r[k] if r else None) for _, k in METRICS])
    rows.append(["SOS official test: published", M["pub_src"], "", "", "", "", fmt("f1_oil", M["published"]["f1_oil"]), "", "",
                 fmt("miou", M["published"]["miou"]), "", "", "", "", "", "", ""])
    D.table("All pooled metrics (%, except ROC-AUC, PR-AUC, ECE, MSE and R²). The SOS leaky-split model is tested on the official "
            "test set, the leak-free model on the leak-free test set (each model's own test set).", rows,
            widths=[4.0, 2.3, 1.1] + [1.28] * 13 + [1.75])
    D.p("**MSE and R².** MSE is the mean squared error between the predicted oil probability and the true 0/1 label of every "
        "pixel (the Brier score). R² = 1 − MSE / Var(label), where Var(label) = f(1 − f) and f is the oil share of the test pixels. "
        "R² = 1 is perfect, 0 equals always predicting the average oil share, and a negative value is worse than that. "
        "These are regression metrics, so they are computed on the probabilities, not on the thresholded masks.")

    # ---------- T: reproduction
    D.h("2. Reproduction check (SOS official test, TTA)", 1)
    t = summ(runs["leaky"], "sentinel", "tta")
    rows = [["Metric", "Ours", "Published", "Gap (pp)", "Within ±2 pp"]]
    for lab, k in [("mIoU", "miou"), ("F1 (oil)", "f1_oil")]:
        g = 100 * (float(t[k]) - M["published"][k])
        rows.append([lab, fmt(k, t[k]), fmt(k, M["published"][k]), f"{g:+.2f}", "yes" if abs(g) <= 2 else "no"])
    D.table(f"Ours vs published ({M['pub_src']}).", rows, widths=[6, 5, 5, 5, 5.5])

    # ---------- T: leaky vs leak-free
    D.h("3. Leaky vs leak-free SOS split (TTA)", 1)
    a, b = J(RUNS, runs["leaky"], "metrics_sentinel.json"), J(RUNS, runs["leakfree"], "metrics_sentinel_groupsplit.json")
    for d_ in (a, b):
        d_["tta"]["r2"] = r2_of(d_["tta"]["brier"], d_["tta"]["oil_pixel_fraction"])
    rows = [["Metric", "Leaky split (official test)", "Leak-free split (leak-free test)", "Difference (leak-free − leaky)"]]
    for lab, k in METRICS:
        x, y = a["tta"][k], b["tta"][k]
        d = f"{100 * (y - x):+.2f} pp" if k in PCT else f"{y - x:+.4f}"
        rows.append([lab, fmt(k, x), fmt(k, y), d])
    rows.append(["Oil share of test pixels", fmt("iou_oil", a["tta"]["oil_pixel_fraction"]), fmt("iou_oil", b["tta"]["oil_pixel_fraction"]),
                 f"{100 * (b['tta']['oil_pixel_fraction'] - a['tta']['oil_pixel_fraction']):+.2f} pp"])
    rows.append(["Test tiles", str(a["n_tiles"]), str(b["n_tiles"]), ""])
    D.table("Same model and recipe; only the SOS split changes.", rows, widths=[7, 6, 6.5, 7])

    # ---------- T: training details + per tile
    D.h("4. Training details and per-tile distribution", 1)
    rows = [["Model", "Training tiles", "Epochs", "Training time", "Evaluated checkpoint", "Its training loss", "Final-epoch loss",
             "Parameters", "Oil test tiles", "Per-tile IoU median", "Tiles with IoU < 0.3"]]
    for mk, name in MODELS:
        log = list(csv.DictReader(open(os.path.join(RUNS, runs[mk], "train_log.csv"))))
        losses = [float(r["train_loss"]) for r in log]
        if M["key"] == "M1":
            ck = f"best training loss (epoch {int(log[losses.index(min(losses))]['epoch'])})"; ckl = min(losses)
        else:
            ck = "final epoch"; ckl = losses[-1]
        tag = "sentinel" if mk == "leaky" else "sentinel_groupsplit"
        pt = [r for r in csv.DictReader(open(os.path.join(RUNS, runs[mk], f"per_tile_{tag}.csv"))) if int(r["gt_oil_px"]) > 0]
        v = [float(r["iou_oil"]) for r in pt]
        sec = int(log[-1]["seconds"]); ntr = AN["train_vs_test"][f"{M['key']}_{'official' if mk == 'leaky' else 'groupsplit'}"]["train"]["n_tiles"]
        mj = J(RUNS, runs[mk], f"metrics_{tag}.json")
        rows.append([name, str(ntr), str(len(log)), f"{sec // 3600} h {sec % 3600 // 60:02d} min", ck, f"{ckl:.4f}", f"{losses[-1]:.4f}",
                     f"{mj['params_M']:.2f} M", str(len(v)), f"{statistics.median(v):.3f}", f"{100 * sum(x < 0.3 for x in v) / len(v):.1f} %"])
    D.table("Training and per-tile oil IoU (TTA; tiles that contain oil).", rows,
            widths=[4.2, 2.0, 1.4, 2.2, 4.0, 2.0, 2.0, 2.0, 2.0, 2.3, 2.4])
    D.p(f"Setup: {M['setup']}.")

    # ---------- T: overfitting
    D.h("5. Overfitting check: training tiles vs test tiles (single pass, no augmentation)", 1)
    rows = [["Model", "Data"] + SHORT[:8] + ["ROC-AUC", "ECE", "MSE", "Loss (BCE+Dice)"]]
    keys = [k for _, k in METRICS[:8]] + ["roc_auc", "ece", "brier", "loss"]
    for mk, name in MODELS:
        T = AN["train_vs_test"][f"{M['key']}_{'official' if mk == 'leaky' else 'groupsplit'}"]
        for part in ("train", "test"):
            rows.append([name, f"{part} ({T[part]['n_tiles']} tiles)"] + [fmt(k, T[part][k]) for k in keys])
        rows.append(["", "gap (train − test)"] + [(f"{100 * (T['train'][k] - T['test'][k]):+.2f}" if k in PCT else f"{T['train'][k] - T['test'][k]:+.4f}") for k in keys])
    D.table("Each model scored on its own training tiles and its test tiles. A large positive gap means overfitting.", rows,
            widths=[4.6, 3.0] + [1.55] * 11 + [2.0])

    # ---------- T: confidence intervals
    D.h("6. 95 % confidence intervals (bootstrap)", 1)
    rows = [["Test", "Model", "F1 (oil)", "IoU (oil)", "mIoU", "Precision", "Recall", "Look-alike called oil", "MDOD AUC oil vs look-alikes"]]
    ci = lambda d: f"{100 * d['value']:.2f} [{100 * d['lo']:.2f}, {100 * d['hi']:.2f}]"  # noqa: E731
    for mk, name in MODELS:
        k = f"{M['key']}_{'official' if mk == 'leaky' else 'groupsplit'}"
        s = AN["sos_ci"][k]
        rows.append(["SOS (own test set), by tile", name] + [ci(s[x]) for x in ("f1_oil", "iou_oil", "miou", "precision_oil", "recall_oil")] + ["—", "—"])
    for mk, name in MODELS:
        k = f"{M['key']}_{'official' if mk == 'leaky' else 'groupsplit'}"
        p = AN["peru_ci"][k]; m = AN["mdod_auc_ci"][k]
        rows.append(["Peru all (by scene) / MDOD (by box)", name] + [ci(p[x]) for x in ("f1_oil", "iou_oil", "miou", "precision_oil", "recall_oil")]
                    + [ci(p["lookalike_called_oil"]), f"{m['value']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"])
    D.table("Value [2.5 %, 97.5 %] over 2,000 resamples (MDOD: 500). SOS resamples tiles; Peru resamples whole scenes "
            "(40), because tiles of one scene are not independent. TTA predictions.", rows,
            widths=[4.6, 4.0, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.7])
    k1, k2 = f"{M['key']}_official", f"{M['key']}_groupsplit"
    dd = AN["peru_ci"][f"{k1}_vs_{k2}"]
    rows = [["Peru metric", "Leaky-trained", "Leak-free-trained", "Difference (leak-free − leaky) [95 % CI]", "Clear difference?"]]
    for lab, x in [("F1 (oil)", "f1_oil"), ("IoU (oil)", "iou_oil"), ("Precision (oil)", "precision_oil"), ("Recall (oil)", "recall_oil")]:
        d = dd[x]["diff_b_minus_a"]
        rows.append([lab, f"{100 * dd[x]['value']:.2f}", f"{100 * (dd[x]['value'] + d['value']):.2f}",
                     f"{100 * d['value']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] pp", "yes" if d["lo"] > 0 or d["hi"] < 0 else "no (CI includes 0)"])
    D.table("Paired comparison on the same Peru patches (scene bootstrap).", rows, widths=[4, 4, 4, 9, 5.5])

    # ---------- T: Peru per class
    D.h("7. Peru: what is called oil, per true class (TTA)", 1)
    pe = {mk: J(RUNS, runs[mk], "peru", "metrics_peru.json") for mk, _ in MODELS}
    rows = [["True class", "Share of pixels (all)", "Called oil: leaky-trained (all)", "Called oil: leak-free-trained (all)",
             "Share of pixels (test subset)", "Called oil: leaky-trained (test subset)", "Called oil: leak-free-trained (test subset)"]]
    for c in PERU_CLASSES:
        rows.append([c, base.p0(pe["leaky"]["all"]["class_pixel_share"][c]), base.p0(pe["leaky"]["all"]["called_oil_rate_per_class"][c]),
                     base.p0(pe["leakfree"]["all"]["called_oil_rate_per_class"][c]), base.p0(pe["leaky"]["official_test"]["class_pixel_share"][c]),
                     base.p0(pe["leaky"]["official_test"]["called_oil_rate_per_class"][c]), base.p0(pe["leakfree"]["official_test"]["called_oil_rate_per_class"][c])])
    D.table("Share of each true Peru class predicted as oil (%). For “oil” this is the recall.", rows, widths=[3, 3.6, 3.9, 4.1, 3.6, 4.1, 4.2])

    # ---------- T: MDOD
    D.h("8. MDOD: box-level alerts (single pass, 400 boxes per class)", 1)
    md = {mk: J(RUNS, runs[mk], "mdod", "metrics_mdod.json") for mk, _ in MODELS}
    rows = [["Class", "Model", "Boxes", "Alert @2 %", "Alert @10 %", "Alert @20 %", "Alert @50 %", "Mean p(oil)", "Median coverage",
             "Mean peak p(oil)", "AUC oil vs class (mean)", "AUC oil vs class (peak)"]]
    for c in CLASSES:
        for mk, name in MODELS:
            q = md[mk]["per_class"][c]
            au = md[mk]["auc_oil_vs"].get(c); ap = md[mk]["auc_oil_vs_peak"].get(c)
            rows.append([c if mk == "leaky" else "", "leaky-trained" if mk == "leaky" else "leak-free-trained", str(q["n_boxes"]),
                         base.p0(q["alert_rate@0.02"]), base.p0(q["alert_rate@0.1"]), base.p0(q["alert_rate@0.2"]), base.p0(q["alert_rate@0.5"]),
                         f"{q['mean_box_score']:.3f}", f"{q['median_coverage']:.3f}", f"{q['mean_peak_score']:.3f}",
                         "—" if au is None else f"{au:.3f}", "—" if ap is None else f"{ap:.3f}"])
    for mk, name in MODELS:
        rows.append(["all look-alikes", "leaky-trained" if mk == "leaky" else "leak-free-trained", "1600", "", "", "", "", "", "", "",
                     f"{md[mk]['auc_oil_vs']['all look-alikes']:.3f}", f"{md[mk]['auc_oil_vs_peak']['all look-alikes']:.3f}"])
    D.table("Alert @x % = share of boxes with at least x % of pixels at p(oil) > 0.5 (%). For oil spills high is good; for "
            "look-alikes low is good. AUC below 0.5 = look-alikes score higher than oil.", rows,
            widths=[2.8, 2.8, 1.3, 1.9, 1.9, 1.9, 1.9, 2.0, 2.1, 2.2, 2.8, 2.8])
    weak_section(D, M)
    path = os.path.join(OUT, M["file"]); D.d.save(path); print("wrote", path, D.nt, "tables")


def analysis_doc():
    D = landscape_doc("Analysis: overfitting, datasets and what the metrics mean", "Analysis of M1 and M2")
    s = D.d.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run("M1 CBD-Net and M2 Improved DeepLabV3+"); r.font.color.rgb = base.INK2
    sd, pd_, mdd = AN["sos_dataset"], AN["peru_dataset"], AN["mdod_dataset"]
    tv = AN["train_vs_test"]
    D.h("1. Are the datasets a problem?", 1)
    o, g = sd["official"], sd["group"]
    D.table("SOS Sentinel-1: tiles vs independent sea areas.", [
        ["Split", "Train tiles", "Test tiles", "Independent sea areas (train)", "Independent sea areas (test)", "Areas in both train and test",
         "Oil share train", "Oil share test"],
        ["Official (leaky)", str(o["train_tiles"]), str(o["test_tiles"]), str(o["train_groups"]), str(o["test_groups"]),
         f"{o['shared_groups']} (all test areas)" if o["shared_groups"] == o["test_groups"] else str(o["shared_groups"]),
         base.p0(o["oil_share_train"]) + " %", base.p0(o["oil_share_test"]) + " %"],
        ["Group-disjoint (leak-free)", str(g["tiles"].get("train", 0) + g["tiles"].get("val", 0) + g["tiles"].get("calib", 0)), str(g["tiles"]["test"]),
         str(g["groups"].get("train", 0) + g["groups"].get("val", 0) + g["groups"].get("calib", 0)), str(g["groups"]["test"]),
         str(g["shared_groups_train_test"]), base.p0(g["oil_share_train"]) + " %", base.p0(g["oil_share_test"]) + " %"],
    ], widths=[4.2, 2.4, 2.4, 3.6, 3.6, 3.6, 3.0, 3.0])
    D.bullets([
        f"**SOS is much smaller than its tile count.** {o['train_tiles'] + o['test_tiles']:,} Sentinel-1 tiles come from only "
        f"{sd['total_groups']:,} independent sea areas (an area appears up to 5 times, as augmented copies). The effective "
        "training set is therefore about a third of its nominal size, which raises the risk of memorising areas.",
        "**In the official split, every test area also appears in training**, so official test scores partly measure "
        "recognition of seen areas (the leak). The group split removes this.",
        f"**The leak-free test set has less oil** ({base.p0(g['oil_share_test'])} % vs {base.p0(o['oil_share_test'])} % of pixels), "
        "which raises background IoU, mIoU and accuracy by itself. Compare models on oil IoU and oil F1.",
    ])
    D.table("Peru and MDOD (test only, never used for training).", [
        ["Dataset", "Size", "Balance check", "Effect on our numbers"],
        ["Peru", f"{pd_['n_patches']:,} patches, {pd_['n_scenes']} scenes (median {pd_['patches_per_scene_median']:.0f}, "
                 f"min {pd_['patches_per_scene_min']}, max {pd_['patches_per_scene_max']} patches per scene)",
         f"largest scene = {100 * pd_['largest_scene_share']:.1f} % of all patches",
         "pooled metrics are weighted towards large scenes; we therefore bootstrap by scene, so the confidence intervals reflect 40 scenes, not 2,013 patches"],
        ["MDOD", "; ".join(f"{c}: {mdd[c]['in_release']:,} boxes" for c in CLASSES),
         "we sample 400 boxes per class (seed 0), so every class has equal weight",
         "; ".join(f"{c} median short side {mdd[c]['median_short_side_px']:.0f} px" for c in CLASSES)],
    ], widths=[2.5, 9.0, 6.0, 9.0], bold_first_col=True)
    D.bullets([
        "**The large datasets are not a problem for training: they are never trained on.** They only test. Their size gives "
        "tighter confidence intervals. What matters is balance, which is handled by sampling equal numbers of boxes per class "
        "(MDOD) and by resampling whole scenes (Peru).",
        "**MDOD differs in scale:** its scenes are about 10× coarser than SOS tiles, so oil boxes contain thin streaks. "
        "This lowers the oil alert rate for both models and is a property of the dataset, not of overfitting.",
        "**MDOD also differs in brightness** (" + "pixel brightness on a random sample (SOS 150 tiles, Peru 150 patches, MDOD 25 scenes, land excluded): mean / median SOS 96 / 89, Peru 88 / 85, MDOD 135 / 142" + "). Peru matches SOS closely, so the Peru failure is not a "
        "brightness effect; MDOD mixes three effects (brightness, resolution, box size) and is therefore reported as a "
        "supplementary look-alike stress test, not as clean evidence.",
    ])

    D.h("2. Do the models overfit?", 1)
    rows = [["Model", "Train IoU oil", "Test IoU oil", "Gap", "Train F1", "Test F1", "Gap", "Train loss", "Test loss", "Loss ratio", "Peru F1 (new sea)"]]
    for key, name in [("M1_official", "M1, leaky-trained"), ("M1_groupsplit", "M1, leak-free-trained"),
                      ("M2_official", "M2, leaky-trained"), ("M2_groupsplit", "M2, leak-free-trained")]:
        T = tv[key]; pf = AN["peru_ci"][key]["f1_oil"]["value"]
        rows.append([name, base.p1(T["train"]["iou_oil"]), base.p1(T["test"]["iou_oil"]), f"{100 * (T['train']['iou_oil'] - T['test']['iou_oil']):+.2f}",
                     base.p1(T["train"]["f1_oil"]), base.p1(T["test"]["f1_oil"]), f"{100 * (T['train']['f1_oil'] - T['test']['f1_oil']):+.2f}",
                     f"{T['train']['loss']:.4f}", f"{T['test']['loss']:.4f}", f"{T['test']['loss'] / T['train']['loss']:.2f}", base.p1(pf)])
    D.table("Train vs test (single pass, no augmentation) and the external test. Train = the model's own training tiles.", rows,
            widths=[4.4, 2.2, 2.2, 1.8, 2.2, 2.2, 1.8, 2.3, 2.3, 2.1, 2.5])
    D.p("**Reading this table:** a small train–test gap means the model generalises to new tiles from the same kind of "
        "data. A large drop from the SOS test to Peru means it does not generalise to a new sea. These are two different "
        "things: the first is classic overfitting, the second is domain shift.")
    g_ = lambda k, m: 100 * (tv[k]["train"][m] - tv[k]["test"][m])  # noqa: E731
    lr_ = lambda k: tv[k]["test"]["loss"] / tv[k]["train"]["loss"]  # noqa: E731
    D.bullets([
        f"**M1 shows almost no overfitting.** Oil IoU gap {g_('M1_official', 'iou_oil'):+.1f} pp on the leaky split and "
        f"{g_('M1_groupsplit', 'iou_oil'):+.1f} pp on the leak-free split. Its schedule stops when the training loss stops improving.",
        f"**The near-zero gap of M1 on the official split is itself a symptom of the leak:** the official test tiles are "
        "augmented copies of training areas, so the test set behaves like the training set. On the leak-free split a gap appears.",
        f"**M2 overfits mildly.** Oil IoU gap {g_('M2_official', 'iou_oil'):+.1f} pp (leaky) and {g_('M2_groupsplit', 'iou_oil'):+.1f} pp "
        f"(leak-free); the test loss is {100 * (min(lr_('M2_official'), lr_('M2_groupsplit')) - 1):.0f}–{100 * (max(lr_('M2_official'), lr_('M2_groupsplit')) - 1):.0f} % above the "
        "training loss. M2 trains for the full 80 epochs with no early stopping, and the last epoch is evaluated (its paper uses no validation set).",
        "**Removing the leak makes the gap larger for both models** (oil IoU gap M1: {:+.1f} → {:+.1f} pp, M2: {:+.1f} → {:+.1f} pp), which confirms that the "
        "official split hides part of the generalisation gap.".format(
            g_('M1_official', 'iou_oil'), g_('M1_groupsplit', 'iou_oil'), g_('M2_official', 'iou_oil'), g_('M2_groupsplit', 'iou_oil')),
        "**Overall, overfitting is small (under 4 pp) compared with the domain shift:** oil F1 falls from about 87 % on SOS "
        "to 22–29 % on Peru. The models are not memorising SOS tiles; they learned a rule (“dark area = oil”) that does not "
        "hold on other seas. The ranking of the effects is: domain shift (≈ 60 pp) ≫ overfitting (≤ 4 pp) > leak (≈ 1–2 pp).",
        "Caveat: training-tile scores are computed without augmentation, and the SOS training set itself contains augmented copies.",
    ])

    D.h("3. Are the differences real? (confidence intervals)", 1)
    rows = [["Comparison", "Test set", "Metric", "Difference [95 % CI]", "Clear?"]]
    for a, b, name in [("M1_official", "M2_official", "M2 − M1 (leaky-trained)"), ("M1_groupsplit", "M2_groupsplit", "M2 − M1 (leak-free-trained)")]:
        d = AN["sos_ci"][f"{a}_vs_{b}"]
        for x, lab in [("f1_oil", "F1"), ("iou_oil", "IoU oil"), ("miou", "mIoU")]:
            q = d[x]["diff_b_minus_a"]
            rows.append([name, "SOS (same tiles)", lab, f"{100 * q['value']:+.2f} [{100 * q['lo']:+.2f}, {100 * q['hi']:+.2f}] pp",
                         "yes" if q["lo"] > 0 or q["hi"] < 0 else "no"])
    for a, b, name in [("M1_official", "M1_groupsplit", "M1: leak-free − leaky"), ("M2_official", "M2_groupsplit", "M2: leak-free − leaky"),
                       ("M1_official", "M2_official", "M2 − M1 (leaky-trained)"), ("M1_groupsplit", "M2_groupsplit", "M2 − M1 (leak-free-trained)")]:
        q = AN["peru_ci"][f"{a}_vs_{b}"]["f1_oil"]["diff_b_minus_a"]
        rows.append([name, "Peru (same patches, by scene)", "F1", f"{100 * q['value']:+.2f} [{100 * q['lo']:+.2f}, {100 * q['hi']:+.2f}] pp",
                     "yes" if q["lo"] > 0 or q["hi"] < 0 else "no"])
    D.table("Paired bootstrap differences. “Clear” = the 95 % interval excludes zero. One training seed per model, so "
            "seed-to-seed variation is not included and the true uncertainty is larger.", rows, widths=[6, 5.5, 2.5, 8, 3])
    sci = AN["sos_ci"]; pci = AN["peru_ci"]; mci = AN["mdod_auc_ci"]
    q = lambda d: f"{100 * d['value']:+.2f} pp [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"  # noqa: E731
    D.bullets([
        f"**SOS, M2 vs M1:** on the leaky split the difference is not clear (F1 {q(sci['M1_official_vs_M2_official']['f1_oil']['diff_b_minus_a'])}); "
        f"on the leak-free split M2 is slightly but clearly better (F1 {q(sci['M1_groupsplit_vs_M2_groupsplit']['f1_oil']['diff_b_minus_a'])}).",
        f"**Peru, leaky vs leak-free training:** clear for M1 (F1 {q(pci['M1_official_vs_M1_groupsplit']['f1_oil']['diff_b_minus_a'])}), "
        f"not clear for M2 ({q(pci['M2_official_vs_M2_groupsplit']['f1_oil']['diff_b_minus_a'])}). So removing the leak "
        "does not consistently improve transfer to a new sea.",
        f"**Peru, M2 vs M1:** M1 is clearly better on the new sea (leak-free models: {q(pci['M1_groupsplit_vs_M2_groupsplit']['f1_oil']['diff_b_minus_a'])}), "
        "the opposite of the SOS ranking. In-domain rankings do not predict out-of-domain rankings.",
        f"**Peru absolute values are very uncertain:** e.g. M1 leaky-trained F1 {100 * pci['M1_official']['f1_oil']['value']:.1f} % "
        f"[{100 * pci['M1_official']['f1_oil']['lo']:.1f}, {100 * pci['M1_official']['f1_oil']['hi']:.1f}]. Performance depends "
        "strongly on the scene, so a few scenes decide the score. Paired differences are much tighter than absolute values.",
        f"**MDOD:** every AUC interval lies far below 0.5 (highest upper bound {max(v['hi'] for v in mci.values()):.2f}), so "
        "“look-alikes score higher than oil” is not chance.",
        "All intervals cover only test-set sampling. Each model is one training seed; seed-to-seed variation is extra.",
    ])
    D.h("4. What the metrics tell us", 1)
    md_ = AN["mdod_dataset"]
    D.bullets([
        "**mIoU can mislead.** It averages oil IoU with background IoU, and background is about 70 % of the pixels. When the "
        "test set has less oil, background IoU rises and mIoU rises even if oil detection gets worse (leak-free split: oil IoU "
        "down, mIoU up). Oil IoU and oil F1 are the honest headline numbers; mIoU is reported because the literature uses it.",
        "**Accuracy is not informative here.** A model that predicts “no oil” everywhere scores about 70 % on SOS and about 94 % on Peru.",
        "**Precision vs recall tells the failure mode.** On Peru recall stays about 95 % but precision falls to 13–17 %: the "
        "models miss little oil but raise about 6 false alarms per true one. Operationally that is the costly error.",
        "**F1 vs IoU:** they always rank models the same way (IoU = F1 / (2 − F1)); IoU is the stricter number.",
        "**ROC-AUC vs ECE:** ROC-AUC stays high on Peru (≈ 0.91–0.93), meaning the probabilities still rank pixels sensibly, while "
        "ECE rises from ≈ 0.03 to 0.19–0.30, meaning the probabilities are no longer trustworthy as probabilities. That combination "
        "(good ranking, bad calibration) is exactly what calibration and threshold methods can fix.",
        "**MSE and R² (regression view of the probabilities):** on SOS R² is clearly positive, but on Peru it is strongly "
        "negative: the over-confident probabilities are worse than predicting the average oil share for every pixel. This is "
        "the same message as ECE.",
        "**Boundary F1 is low (38–48 %)** because SOS labels are coarse polygons; it measures label agreement on edges as much as model quality.",
        "**The official-script F1 (93–95 %) is a background-class score** and should never be quoted as oil F1.",
        f"**MDOD scores are confounded by box size:** oil boxes are small (median short side {md_['oil spill']['median_short_side_px']:.0f} px) "
        f"while look-alike boxes are large ({md_['rain cell']['median_short_side_px']:.0f}–{md_['biogenic slick']['median_short_side_px']:.0f} px). "
        "Mean-score and coverage comparisons between classes are therefore partly comparisons of box size; the alert rate "
        "and AUC should be read as “how often the model raises an alarm on each phenomenon”, not as pixel accuracy.",
    ])
    glossary(D)
    path = os.path.join(OUT, "Analysis_Overfitting_Datasets_Metrics.docx"); D.d.save(path); print("wrote", path)
    return D


# ======================= weaknesses (critique) =======================
TILE_PX = 256 * 256
LOOKALIKES = ["biogenic slick", "low wind", "rain cell", "sea ice"]
SIZE_BINS = [("small spill (< 2 % of tile)", 0, 0.02), ("medium (2–20 %)", 0.02, 0.20), ("large (≥ 20 %)", 0.20, 1.01)]


def wdata(M):
    """Weak-point numbers for one method, from the saved metric files and per-tile CSVs (no model is run)."""
    out = {}
    for mk in ("leaky", "leakfree"):
        run = M["runs"][mk]; tag = "sentinel" if mk == "leaky" else "sentinel_groupsplit"
        s = J(RUNS, run, f"metrics_{tag}.json")["tta"]
        pe = J(RUNS, run, "peru", "metrics_peru.json")["all"]
        md = J(RUNS, run, "mdod", "metrics_mdod.json")
        pt = list(csv.DictReader(open(os.path.join(RUNS, run, f"per_tile_{tag}.csv"))))
        oil = [r for r in pt if int(r["gt_oil_px"]) > 0]; empty = [r for r in pt if int(r["gt_oil_px"]) == 0]
        bins = {}
        for lab, lo, hi in SIZE_BINS:
            v = [float(r["iou_oil"]) for r in oil if lo <= int(r["gt_oil_px"]) / TILE_PX < hi]
            bins[lab] = dict(n=len(v), mean=statistics.mean(v) if v else None, median=statistics.median(v) if v else None,
                             bad=sum(x < 0.3 for x in v) / len(v) if v else None)
        fa = [r for r in empty if int(r["pred_oil_px"]) >= 0.01 * TILE_PX]
        out[mk] = dict(
            sos_iou=s["iou_oil"], sos_recall=s["recall_oil"], sos_bf1=s["boundary_f1"],
            sos_small=bins[SIZE_BINS[0][0]]["mean"], sos_bad=sum(float(r["iou_oil"]) < 0.3 for r in oil) / len(oil),
            sos_empty_fa=(len(fa) / len(empty)) if empty else None, n_empty=len(empty), bins=bins,
            peru_p=pe["precision_oil"], peru_f1=pe["f1_oil"], peru_iou=pe["iou_oil"], peru_la=pe["called_oil_rate_per_class"]["look-alike"],
            peru_ece=pe["ece"], peru_brier=pe["brier"], peru_prauc=pe["pr_auc"],
            peru_r2=r2_of(pe["brier"], pe["oil_pixel_fraction"]),
            mdod_oil=md["per_class"]["oil spill"]["alert_rate@0.02"],
            mdod_la=statistics.mean(md["per_class"][c]["alert_rate@0.02"] for c in LOOKALIKES),
            mdod_auc=md["auc_oil_vs"]["all look-alikes"])
    return out


# (key, label, higher_is_better, kind, why it is weak)   kind: pct = percentage metric (1 pp target), abs = 0.01 target
WEAK = [
    ("sos_iou", "SOS oil IoU (each model's own test set)", True, "pct", "Headline in-domain number; the leak-free value is the honest one."),
    ("sos_recall", "SOS oil recall", True, "pct", "Missed oil; falls when the leak is removed."),
    ("sos_bf1", "SOS boundary F1", True, "pct", "Slick outlines are poor (partly coarse polygon labels)."),
    ("sos_small", "SOS small-spill tiles: mean IoU", True, "pct", "Thin / small slicks are the hardest tiles."),
    ("sos_bad", "SOS oil tiles with IoU < 0.3", False, "pct", "Share of tiles that are essentially failed."),
    ("peru_p", "Peru precision (oil)", True, "pct", "About 6 false-alarm pixels per true oil pixel on a new sea."),
    ("peru_f1", "Peru F1 (oil)", True, "pct", "Transfer to a new sea collapses (≈ 87 % → 22–29 %)."),
    ("peru_iou", "Peru IoU (oil)", True, "pct", "Same failure, stricter metric."),
    ("peru_la", "Peru look-alike pixels called oil", False, "pct", "The core failure: SOS has no look-alike class."),
    ("peru_prauc", "Peru PR-AUC", True, "abs", "Ranking of oil vs everything else on the new sea."),
    ("peru_ece", "Peru ECE (calibration)", False, "abs", "Probabilities are over-confident (≈ 0.03 on SOS)."),
    ("peru_brier", "Peru MSE (= Brier score)", False, "abs", "Probability error, combines calibration and accuracy."),
    ("peru_r2", "Peru R² (probabilities vs labels)", True, "abs", "Negative: worse than predicting the average oil share everywhere."),
    ("mdod_oil", "MDOD oil boxes alerted (@ 2 %)", True, "pct", "Many real oil boxes are missed at MDOD's coarser resolution."),
    ("mdod_la", "MDOD look-alike boxes alerted (@ 2 %, mean of 4 classes)", False, "pct", "Almost every look-alike raises an alarm."),
    ("mdod_auc", "MDOD AUC, oil vs look-alikes", True, "abs", "Below 0.5: look-alikes score higher than oil."),
]


def wfmt(kind, v):
    if v is None:
        return "—"
    return f"{100 * v:.2f}" if kind == "pct" else f"{v:.3f}"


def target(kind, hib, v):
    """What a contribution must reach to beat this baseline by at least 1 pp (or 0.01 for scores)."""
    if v is None:
        return "—"
    return (">" if hib else "<") + " " + wfmt(kind, v + 0.01 if hib else v - 0.01)


def weak_section(D, M):
    w = wdata(M)
    D.h("9. Where the method is weak (critique) and what a contribution must beat", 1)
    D.p("Each row is a metric on which the method performs badly. **Baseline for contributions = the leak-free-trained "
        "model** (the fair one). A contribution must beat it by at least 1 pp (0.01 for PR-AUC, ECE, MSE, R² and AUC). "
        "SOS rows use each model's own test set. Single seed, so differences under about 1 pp on SOS are within noise "
        "(see section 6).")
    rows = [["Metric", "Better", "Leaky-trained", "Leak-free-trained", "Target (beat leak-free by ≥ 1 pp)", "Why it is weak"]]
    for k, lab, hib, kind, why in WEAK:
        rows.append([lab, "higher" if hib else "lower", wfmt(kind, w["leaky"][k]), wfmt(kind, w["leakfree"][k]),
                     target(kind, hib, w["leakfree"][k]), why])
    D.table("Weak points (%, except PR-AUC, ECE, MSE, R² and AUC). Computed from the saved predictions; no model was re-run.",
            rows, widths=[6.4, 1.5, 2.3, 2.6, 3.6, 10.1])
    rows = [["Spill size in tile", "Model", "Oil tiles", "Mean IoU", "Median IoU", "Tiles with IoU < 0.3"]]
    for lab, _, _ in SIZE_BINS:
        for mk, name in MODELS:
            b = w[mk]["bins"][lab]
            rows.append([lab if mk == "leaky" else "", "leaky-trained" if mk == "leaky" else "leak-free-trained", str(b["n"]),
                         wfmt("pct", b["mean"]), wfmt("pct", b["median"]), wfmt("pct", b["bad"])])
    D.table("SOS per-tile oil IoU by spill size (share of the 256 × 256 tile covered by true oil; TTA).", rows,
            widths=[5.5, 4, 3, 4, 4, 5.5])
    D.p(f"Oil-free test tiles: {w['leaky']['n_empty']} (official test) and {w['leakfree']['n_empty']} (leak-free test). "
        f"Share with a false alarm covering at least 1 % of the tile: {wfmt('pct', w['leaky']['sos_empty_fa'])} % and "
        f"{wfmt('pct', w['leakfree']['sos_empty_fa'])} %, so clean SOS sea is not a weakness; the false alarms appear on "
        "look-alikes in other seas.")


# ======================= professor's questions =======================
def qa_doc():
    D = landscape_doc("Datasets, implementation checks and critique of M1 and M2", "Datasets, checks and critique")
    s = D.d.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run("Answers to the review questions  ·  M1 CBD-Net, M2 Improved DeepLabV3+, M3 FCS-Net (coded, not trained)"
                  ""); r.font.color.rgb = base.INK2
    sd, pd_, md_ = AN["sos_dataset"], AN["peru_dataset"], AN["mdod_dataset"]
    pe = J(RUNS, METHODS["M1"]["runs"]["leaky"], "peru", "metrics_peru.json")["all"]["class_pixel_share"]

    # ---- Q1
    D.h("1. Shape of the datasets and what they contain", 1)
    rows = [["Dataset", "Sensor", "Images", "Image size", "Format", "Labels", "Class balance", "Our use"],
            ["SOS [P24]", "Sentinel-1 (C-band)", f"{sd['official']['train_tiles']:,} train / {sd['official']['test_tiles']:,} test tiles",
             "256 × 256", "JPG image (3 channels) + PNG mask", "binary pixel mask: 255 = oil, 0 = background",
             f"oil {100 * sd['official']['oil_share_train']:.1f} % (train) / {100 * sd['official']['oil_share_test']:.1f} % (test) of pixels",
             "training + in-domain test"],
            ["SOS [P24]", "ALOS PALSAR (L-band)", "3,101 train / 776 test tiles", "256 × 256", "same", "same", "—",
             "not used yet (same leak found)"],
            ["Peru [N22]", "Sentinel-1", f"{pd_['n_patches']:,} patches from {pd_['n_scenes']} scenes",
             "512 × 512 (cut into four 256 × 256 tiles)", "GeoTIFF + manifest (scene, date, lat/lon)",
             "5-class pixel mask: oil, look-alike, sea, ship, land", ", ".join(f"{c} {100 * pe[c]:.1f} %" for c in PERU_CLASSES),
             "test only (new sea, look-alikes)"],
            ["MDOD [N05]", "Sentinel-1", "4,673 scenes, 24,940 boxes", "about 2,900 × 1,700", "greyscale image, land = 255; COCO-style boxes",
             "bounding boxes only (no pixel masks)", ", ".join(f"{c} {md_[c]['in_release']:,}" for c in CLASSES),
             "test only (false alerts per look-alike type)"]]
    D.table("Datasets. Network input: batch × 3 × 256 × 256 image; target and output: batch × 1 × 256 × 256 oil mask / probability.",
            rows, widths=[2.0, 2.6, 3.5, 3.0, 3.4, 3.8, 5.0, 3.2])
    D.bullets([
        f"**SOS is smaller than its tile count:** its {sd['official']['train_tiles'] + sd['official']['test_tiles']:,} Sentinel-1 tiles come "
        f"from {sd['total_groups']:,} independent sea areas; each area appears as up to 5 augmented copies (crop, rotation, flip, zoom, noise).",
        "**SOS has no look-alike class:** every dark area in a training tile is either labelled oil or is background sea. The "
        "models are never shown a dark area that is not oil, which explains the Peru and MDOD failures.",
        f"**Peru is scene-structured:** median {pd_['patches_per_scene_median']:.0f} patches per scene "
        f"({pd_['patches_per_scene_min']}–{pd_['patches_per_scene_max']}), so patches of one scene are not independent. "
        "Confidence intervals resample whole scenes.",
        "**MDOD is about 10× coarser than SOS, noticeably brighter, and only has boxes**, so it supports box-level alert tests, "
        "not pixel IoU. It is used as a supplementary look-alike stress test.",
    ])

    D.h("Rows and columns before and after cleaning", 2)
    D.p("For image data a **row is one image (tile)** and the **columns are its pixels × channels**; every pixel is also one "
        "labelled sample. No image was removed from SOS or Peru: “cleaning” here means format conversion, label binarisation, "
        "tiling, and (MDOD) dropping tiny boxes.")
    rows = [["Dataset", "Stage", "Rows (images)", "Columns per image", "Label per image", "Pixels (labelled samples)", "What changed"],
            ["SOS Sentinel-1", "as downloaded", "4,193 (3,354 train + 839 test)", "256 × 256 × 3, uint8 (3 identical greyscale channels)",
             "256 × 256 × 3 PNG, values {0, 255}", "4,193 × 65,536 = 274.8 M", "—"],
            ["", "after cleaning (model input)", "4,193 (none removed; all load)", "3 × 256 × 256 float, scaled to [−1.6, 1.6]",
             "1 × 256 × 256, values {0, 1}", "274.8 M", "channels first; intensity scaling; mask reduced to 1 channel and binarised (≥ 0.5)"],
            ["", "official split (leaky)", "train 3,354 / test 839", "same", "same", "train 219.8 M / test 55.0 M",
             "M3 also holds out 10 % of train for validation: 3,019 / 335 / 839"],
            ["", "leak-free split", "train 2,516 / val 422 / calib 420 / test 835", "same", "same", "train+val+calib 220.1 M / test 54.7 M",
             "rows re-assigned by sea area (1,089 groups, none shared); M1/M2 train on train+val+calib = 3,358"],
            ["Peru", "as downloaded", "2,013 patches (40 scenes)", "512 × 512 × 3, uint8 GeoTIFF", "512 × 512, classes 0–4",
             "2,013 × 262,144 = 527.7 M", "—"],
            ["", "after cleaning (model input)", "8,052 tiles (each patch cut into 4); test subset 338 patches = 1,352 tiles",
             "3 × 256 × 256 float, same scaling", "binary target: 1 = oil, 0 = everything else (5-class label kept for per-class analysis)",
             "527.7 M", "tiling only; none removed; used for testing only"],
            ["MDOD", "as downloaded", "4,673 scenes, 24,940 boxes", "≈ 2,900 × 1,700 × 1 greyscale (land = 255)",
             "boxes: class + normalised x, y, w, h", "—", "—"],
            ["", "after cleaning (model input)", "boxes with short side < 8 px dropped; 400 per class sampled (seed 0) = 2,000 boxes",
             "greyscale repeated to 3 channels; sliding 256 × 256 tiles, overlaps averaged", "box class", "—",
             "no pixel masks, so box-level scores only; test only"]]
    D.table("Dataset size before and after cleaning.", rows, widths=[2.4, 3.2, 4.4, 4.4, 3.8, 3.4, 4.9])

    D.h("Features used in training and testing", 2)
    rows = [["Item", "Training (SOS)", "Testing (SOS, Peru, MDOD)"],
            ["Input feature", "SAR backscatter intensity of every pixel (greyscale 0–255, stored as 3 identical channels)", "the same"],
            ["Values per sample", "3 × 256 × 256 = 196,608 input values per tile (65,536 distinct pixels)", "the same (Peru and MDOD are tiled to 256 × 256)"],
            ["Hand-crafted features", "none: each CNN learns its own features (edges, texture, shape, context) in its encoder", "the same learned features"],
            ["Metadata (wind, incidence angle, location, date)", "not used", "not used (candidate input for the contributions)"],
            ["Target", "per-pixel oil mask, 1 × 256 × 256, values {0, 1}", "SOS: same; Peru: 1 = oil vs the rest; MDOD: box class only"],
            ["Augmentation", "M1: official HSV / shift / scale / flips / 90° rotations; M2, M3: flips and 90° rotations", "none, or 8-way flips/rotations averaged (TTA)"],
            ["Normalisation", "[0, 255] → [−1.6, 1.6] (official CBD-Net scaling, used for all methods)", "the same"]]
    D.table("Features. Training and testing use exactly the same input features and preprocessing.", rows, widths=[5.0, 11.0, 10.5])

    # ---- Q2
    D.h("2. Did we analyse the datasets before using them?", 1)
    rows = [["Check", "What we did", "Finding", "Consequence"],
            ["Train/test independence (SOS)", "Searched every test tile against all training tiles (8 rotations/flips, "
             "normalised cross-correlation); checked the ID structure", "Tiles 20001–22220 = 5 × 444 augmented copies; the official test "
             "set is the first IDs, so 100 % of test tiles (839/839) have copies in training; 571/839 best matches fall exactly on "
             "the copy offsets (chance ≈ 0.1 %)", "Built a group-disjoint (leak-free) split: 2,516 / 422 / 420 / 835 tiles, "
             "no area shared; every model trained and tested on both splits"],
            ["Class balance (SOS)", "Oil share of pixels per split", "29.3 % / 34.7 % (official), 30.6 % / 29.6 % (leak-free)",
             "mIoU and accuracy depend on oil share, so we compare on oil IoU / F1"],
            ["Label meaning (Peru)", "Inspected masks and the dataset description", "1 = oil and 2 = look-alike confirmed; 0 / 3 / 4 "
             "(sea / ship / land) assumed", "Only the oil class is scored as positive; look-alike false alarms reported separately"],
            ["Scene structure (Peru)", "Patches per scene from the manifest", f"{pd_['n_scenes']} scenes, {pd_['patches_per_scene_min']}–"
             f"{pd_['patches_per_scene_max']} patches each; largest scene {100 * pd_['largest_scene_share']:.1f} % of patches",
             "Scene-level bootstrap for confidence intervals"],
            ["Class IDs (MDOD)", "Release has no class names; compared with the paper's class order and example figures",
             "1 biogenic slick, 2 low wind, 3 oil, 4 rain cell, 5 sea ice; the release is larger than the paper (24,940 vs 11,040 objects) "
             "and has no split or pixel masks", "Box-level alert evaluation, 400 boxes per class"],
            ["Box sizes (MDOD)", "Short side of every sampled box", f"oil median {md_['oil spill']['median_short_side_px']:.0f} px; look-alikes "
             f"{md_['rain cell']['median_short_side_px']:.0f}–{md_['biogenic slick']['median_short_side_px']:.0f} px", "Size confound: "
             "class comparisons are partly size comparisons"],
            ["Resolution", "Compared pixel spacing", "MDOD ≈ 10× coarser than SOS", "Thin streaks are only a few pixels wide on MDOD"],
            ["Brightness", "Compared pixel intensity distributions", "pixel brightness on a random sample (SOS 150 tiles, Peru 150 patches, MDOD 25 scenes, land excluded): mean / median SOS 96 / 89, Peru 88 / 85, MDOD 135 / 142",
             "Peru is comparable to SOS (its failure is not brightness); MDOD is a supplementary test only"],
            ["Channels", "Compared the three channels of every sampled image", "SOS and Peru: 3 identical greyscale channels; MDOD: 1 channel",
             "Same input format for all: greyscale repeated to 3 channels"],
            ["Image / mask pairing (SOS)", "Matched every image to its mask by ID", "3,354 / 3,354 train and 839 / 839 test pairs; no ID in both train and test",
             "No missing or orphan files"],
            ["Leak-free split", "Checked every pair of partitions for shared sea areas", "0 shared areas in all 6 pairs; 4,193 unique tiles",
             "The leak-free benchmark is clean"]]
    D.table("Data checks.", rows, widths=[3.6, 6.6, 9.0, 7.3])

    # ---- Q3
    D.h("3. How the implementations work", 1)
    rows = [["Step", "M1 CBD-Net [P24]", "M2 Improved DeepLabV3+ [P01]", "M3 FCS-Net [P29]"],
            ["Code source", "official code, copied unchanged (2 compatibility fixes)", "re-implemented from the paper (no code released)",
             "re-implemented from the paper (no code released)"],
            ["Encoder", "ResNet-34 (ImageNet) + scSE after every stage", "MobileNetV2 (ImageNet), output stride 16, scSE in every block",
             "ConvNeXt-Small (ImageNet-22K) + frequency-spatial coordinate attention after every stage"],
            ["Context module", "dilated block (rates 1, 2, 4, 8)", "ASPP (1×1, rates 6/12/18, image pooling) + scSE",
             "strip-augmented pyramid pooling (1, 2, 3, 6 + strip pooling)"],
            ["Decoder", "LinkNet decoder", "DeepLabV3+ decoder with 1/4-resolution skip", "4 U-Net stages"],
            ["Size", "31.4 M parameters", "6.04 M parameters, 6.56 GMACs (paper: 5.84 M, 6.16)", "65.8 M parameters, 15.7 GMACs (not in paper)"],
            ["Training", "Adam 2e-4, batch 4, ≤ 80 epochs, plateau schedule + early stop, official augmentation, BCE + Dice",
             "Adam 1e-3, batch 15, 80 epochs, BCE + Dice; cosine decay and flips assumed",
             "Adam 1e-3, batch 16 (8 × 2), 80 epochs, BCE + Dice, mixed precision"],
            ["Input scaling", "[0, 255] → [−1.6, 1.6] (official CBD-Net)", "same scaling (assumed; the paper does not state it)",
             "same scaling (assumed; the paper does not state it)"],
            ["Model kept", "best training loss", "final epoch (paper has no validation set)", "best validation mIoU (10 % held out)"],
            ["Evaluation (all)", "threshold 0.5; 8-way test-time augmentation and single pass; metrics pooled over all test pixels; "
             "same evaluation code for every method", "same", "same (pending training)"]]
    D.table("Implementation summary.", rows, widths=[3.0, 7.5, 8.0, 8.0])

    # ---- Q4
    D.h("4. Are the implementations correct? Cross-checks", 1)
    rows = [["Check", "M1", "M2", "M3"]]
    for key, lab in (("miou", "mIoU"), ("f1_oil", "F1")):
        cells = []
        for m in ("M1", "M2"):
            t = summ(METHODS[m]["runs"]["leaky"], "sentinel", "tta")
            cells.append(f"ours {fmt(key, t[key])} vs published {fmt(key, METHODS[m]['published'][key])} "
                         f"({100 * (float(t[key]) - METHODS[m]['published'][key]):+.2f} pp)")
        rows.append([f"Reproduction, SOS official test, {lab}"] + cells + ["target 89.62 mIoU / 92.87 F1; not trained yet"])
    rows += [["Official code vs authors' repository", "byte-identical to github.com/CUG-URS/CBDNet-main (commit ba3aa2e)",
              "no code released", "no code released"],
             ["Model size vs paper", "official code (31.4 M)", "6.04 M vs 5.84 M (+3 %); 6.56 vs 6.16 GMACs", "paper reports none; 65.8 M measured"],
             ["Our metric code vs official script", "Official script reproduced; found it reports the BACKGROUND class "
              "(labels=[255, 0]), so we report oil-class metrics and the script value separately", "same", "same"],
             ["Overfitting check", "train − test oil IoU gap +0.1 / +1.3 pp (leaky / leak-free)", "+1.8 / +3.7 pp", "—"],
             ["Leak-free re-training", "done", "done", "planned"],
             ["Statistical check", "bootstrap 95 % CIs (tile / scene / box) and paired differences", "same", "—"],
             ["Smoke test", "full runs", "full runs", "builds, loads weights, trains, validates, resumes (48-tile run)"],
             ["Assumptions (not in paper)", "none (official code); the paper's boundary loss is not used, as in the released code",
              "cSE reduction 16, 256 ASPP channels, 48-channel skip, cosine decay, flip/rot90, input scaling [−1.6, 1.6] instead of ImageNet mean/std",
              "CA reduction 32, real part in Eq. 2, gate design, SAPP/decoder widths, FS-CA in encoder only, cosine decay, input scaling [−1.6, 1.6]"]]
    D.table("Correctness checks. Both reproductions are within ±2 pp of the published numbers.", rows, widths=[5.0, 7.0, 7.5, 7.0])
    D.bullets([
        "**Limits of the checks:** one training seed per model; M2 and M3 depend on the assumptions listed; the M2 paper does "
        "not say which class its F1 is for.",
        "**M2's published numbers are internally odd:** an oil F1 of 89.62 % implies oil IoU ≈ 81.2 %, so mIoU 81.34 % would need "
        "background IoU ≈ 81.5 %, below oil IoU although background is ≈ 65 % of pixels (we measure 87–90 %). Their F1 and mIoU "
        "were probably computed differently, so the published F1 should not be compared directly.",
    ])

    # ---- Q5
    D.h("5. Metrics on which the methods perform badly (targets for contributions)", 1)
    w = {m: wdata(METHODS[m]) for m in ("M1", "M2")}
    rows = [["Metric", "Better", "M1 leaky", "M1 leak-free", "M2 leaky", "M2 leak-free", "M3", "Best leak-free baseline",
             "Contribution target (≥ 1 pp)", "Why it is weak"]]
    for k, lab, hib, kind, why in WEAK:
        lf = [w[m]["leakfree"][k] for m in ("M1", "M2") if w[m]["leakfree"][k] is not None]
        best = (max(lf) if hib else min(lf)) if lf else None
        rows.append([lab, "higher" if hib else "lower"] + [wfmt(kind, w[m][mk][k]) for m in ("M1", "M2") for mk in ("leaky", "leakfree")]
                    + ["pending", wfmt(kind, best), target(kind, hib, best), why])
    D.table("Weak points of the methods (%, except PR-AUC, ECE, MSE, R², AUC). The target is set against the best leak-free-trained "
            "method, so a contribution that reaches it beats both. M3 = pending training (coded, not trained).", rows,
            widths=[5.0, 1.3, 1.6, 1.8, 1.6, 1.8, 1.5, 2.2, 2.6, 7.1])
    D.bullets([
        "**Largest gaps (most room for a ≥ 1 pp gain):** Peru precision, look-alike pixels called oil, Peru ECE, MDOD AUC, "
        "MDOD look-alike alert rate, SOS boundary F1 and small-spill tiles.",
        "**Smallest gaps (hard to show +1 pp):** SOS oil F1 / IoU. The SOS confidence interval is about ±1 pp, so a +1 pp claim "
        "on SOS needs CIs or several seeds.",
    ])

    D.h("Evaluation protocol the contributions must follow", 2)
    D.bullets([
        "**No SOS data is left for calibration.** The leak-free models were trained on train + val + calib (3,358 tiles), so the "
        "“calib” and “val” partitions are training data. Calibrating or setting alert thresholds on them would use training "
        "data and is not allowed.",
        "**Use Peru's own scene-disjoint split instead:** fit any calibration, threshold or alert rule on Peru train "
        "(1,594 patches, 33 scenes) and report it **only on Peru test** (338 patches, 5 scenes). These share no scene.",
        "**“Peru, all 2,013” includes Peru's train patches.** That is valid for the frozen M1/M2 models (never trained on Peru), "
        "but any result that uses Peru train must be reported on the 338-patch test subset only.",
        "**SOS claims under about 1 pp need more than one seed** (or the confidence intervals), because each model is one training run.",
        "**MDOD is a supplementary stress test** (brightness, resolution and box-size differences), not the main evidence.",
    ])

    # ---- Q6
    D.h("6. Critique of the methods", 1)
    D.bullets([
        "**Leaky benchmark.** All three papers report on the official SOS split, where every test area is also in training. "
        "Removing the leak lowers oil IoU by 1.3–1.8 pp and oil F1 by 0.8–1.1 pp, and doubles M2's overfitting gap.",
        "**No look-alike training data.** SOS labels every dark area as oil or sea, so both methods learn “dark = oil”: "
        "86–90 % of Peru look-alike pixels are called oil and MDOD look-alikes are flagged more often than oil (AUC < 0.5). "
        "Two different architectures fail the same way, so the cause is the data, not the network.",
        "**No cross-sea test.** None of the papers tests on another sea. On Peru oil F1 falls from ≈ 87 % to 22–29 %, and the "
        "ranking reverses (M1 > M2 on Peru, M2 ≥ M1 on SOS).",
        "**No uncertainty or calibration.** No paper reports confidence intervals or calibration, and only M3 reports seeds "
        "(in its ablation). On Peru the probabilities are badly over-confident (ECE 0.19–0.30), so they cannot drive alerts as they are.",
        "**Reproducibility.** M2 and M3 released no code, and key settings (attention reductions, channel widths, schedule, "
        "augmentation) are not given. M3's text contradicts itself about where its attention modules go. M1's paper describes "
        "a boundary loss that its released code does not use by default.",
        "**Architecture claims vs evidence.** On the leak-free split M2 beats M1 by only +0.41 pp F1, and on Peru it is 7 pp "
        "worse. Gains from extra attention modules on SOS do not carry over to new seas.",
        "**Coarse labels.** SOS masks are hand-drawn polygons (see section 7), which caps boundary F1 and makes pixel-exact "
        "metrics partly a measure of label noise.",
    ])
    D.h("Critique of the metrics", 2)
    D.bullets([
        "**mIoU** averages in background IoU (≈ 70 % of pixels); it rose on the leak-free split while oil IoU fell. Oil IoU / F1 should lead.",
        "**Accuracy** is uninformative: predicting “no oil” everywhere gives ≈ 70 % on SOS and ≈ 94 % on Peru.",
        "**The official SOS script's F1 is the background class** (93–95 %); quoting it as oil F1 overstates performance.",
        "**Pooled pixel metrics hide failures:** large spills dominate the pixel count, while small-spill tiles score far lower (section 5).",
        "**Precision and recall must be reported together:** on Peru recall stays ≈ 95 % while precision falls to 13–17 %.",
        "**Probabilities need calibration metrics (ECE, MSE / Brier, R²)**, which the papers do not report; ROC-AUC alone looks fine (≈ 0.91–0.93).",
        "**MDOD box metrics are confounded by size** (oil boxes are much smaller), so they measure alarm rates, not pixel accuracy.",
    ])

    # ---- Q7
    D.h("7. Qualitative analysis", 1)
    D.p("Overlay colours: green = correct oil (TP), red = false alarm (FP), blue = missed oil (FN).")
    figs = [(os.path.join(EXP, "M1_CBDNet", "results", "sentinel_group_split", "qualitative.png"),
             "M1, leak-free SOS test: median tiles (left) and worst tiles (right). Failures: missed thin streaks (blue), a dark "
             "patch called oil (red), and a large labelled polygon that is barely darker than the sea."),
            (os.path.join(EXP, "M2_DeepLabV3Plus", "results", "sentinel_group_split", "qualitative.png"),
             "M2, leak-free SOS test: same layout."),
            (os.path.join(EXP, "M2_DeepLabV3Plus", "results", "peru_external", "qualitative_groupsplit.png"),
             "M2 (leak-free-trained) on Peru. Oil is found (green), but the three look-alike patches (orange label) are almost "
             "entirely called oil (red), and wind streaks near real oil raise false alarms."),
            (os.path.join(EXP, "M1_CBDNet", "results", "peru_external", "qualitative_groupsplit.png"),
             "M1 (leak-free-trained) on Peru: same layout."),
            (os.path.join(EXP, "M1_CBDNet", "results", "mdod_external", "alert_rates.png"), "M1 on MDOD: share of boxes alerted, per class."),
            (os.path.join(EXP, "M2_DeepLabV3Plus", "results", "mdod_external", "alert_rates.png"), "M2 on MDOD: share of boxes alerted, per class.")]
    for p, cap in figs:
        if os.path.exists(p):
            D.fig(p, cap, width_cm=15 if "alert" in p else 24)
    D.h("8. Accuracy, F1, MSE and R² for all three methods", 1)
    plan = [("SOS official test (leaky)", "leaky", "sentinel"), ("SOS leak-free test", "leakfree", "sentinel_groupsplit"),
            ("Peru, all 2,013 patches", "leaky", "peru_all"), ("Peru, all 2,013 patches", "leakfree", "peru_all"),
            ("Peru test subset (338)", "leaky", "peru_official_test"), ("Peru test subset (338)", "leakfree", "peru_official_test")]
    rows = [["Test", "Model trained on", "M1 Acc", "M1 F1", "M1 MSE", "M1 R²", "M2 Acc", "M2 F1", "M2 MSE", "M2 R²",
             "M3 Acc", "M3 F1", "M3 MSE", "M3 R²"]]
    for test, mk, ts in plan:
        cells = []
        for m in ("M1", "M2"):
            r = summ(METHODS[m]["runs"][mk], ts, "tta")
            cells += [fmt("accuracy", r["accuracy"]), fmt("f1_oil", r["f1_oil"]), f"{float(r['brier']):.4f}", f"{float(r['r2']):.3f}"]
        rows.append([test, "leaky split" if mk == "leaky" else "leak-free split"] + cells + ["pending"] * 4)
    rows.append(["SOS official test: published", "leaky split", "—", "87.81", "—", "—", "—", "89.62", "—", "—", "—", "92.87", "—", "—"])
    D.table("Accuracy and F1 in %, MSE and R² on the oil probabilities (TTA). M3 is coded but not trained, so its columns are "
            "pending. The papers report no accuracy, MSE or R².", rows,
            widths=[4.4, 2.6] + [1.45] * 12)
    D.bullets([
        "**Accuracy is 91–93 % on SOS but only 59–73 % on Peru,** below the ≈ 94 % a model would get on Peru by predicting "
        "“no oil” everywhere, because of the many look-alike false alarms. F1 shows the oil performance directly.",
        "**R² is clearly positive on SOS** (the probabilities explain most of the label variance) but **strongly negative on "
        "Peru**: the over-confident probabilities are worse than predicting the average oil share for every pixel. This is the "
        "domain-shift failure expressed as a regression metric.",
        "**MSE / R² are regression metrics:** they are reported because they were requested, computed on the per-pixel "
        "probabilities against the 0/1 labels. MDOD has no pixel labels, so it has no MSE or R².",
    ])

    D.p("All quantitative tables are in M1_CBDNet_All_Results.docx and M2_DeepLabV3Plus_All_Results.docx (same folder); "
        "overfitting, confidence intervals and metric explanations are in Analysis_Overfitting_Datasets_Metrics.docx.")
    path = os.path.join(OUT, "Datasets_Checks_Critique_QA.docx"); D.d.save(path); print("wrote", path)


if __name__ == "__main__":
    for m in METHODS.values():
        method_doc(m)
    analysis_doc()
    qa_doc()
