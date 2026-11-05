"""Render excerpts of the real M1 console logs as terminal-style PNGs for the report.

Lines are copied verbatim from M1_CBDNet/logs/*.log; '...' marks lines left out.
usage (from experiments/):  python M1_CBDNet/report/make_terminal_pics.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(os.path.dirname(HERE), "logs")
OUT = os.path.join(HERE, "terminal"); os.makedirs(OUT, exist_ok=True)
BG, FG, DIM, HI, BAR = "#1e1e1e", "#d4d4d4", "#808080", "#9cdcfe", "#323233"


def read(name):
    return open(os.path.join(LOGS, name), encoding="utf-8", errors="replace").read().splitlines()


def excerpt(lines, head, tail, keep=()):
    """first `head` lines, any line matching `keep`, last `tail` lines, with '...' for gaps."""
    idx = sorted(set(range(head)) | {i for i, l in enumerate(lines) if any(k in l for k in keep)}
                 | set(range(len(lines) - tail, len(lines))))
    out, prev = [], -1
    for i in idx:
        if i != prev + 1:
            out.append("...")
        out.append(lines[i]); prev = i
    return out


def render(lines, title, fname, hl=("[tta]", "[plain]", "[peru", "AUC oil", "done ", "lr ->", "==")):
    width = max(len(l) for l in lines + [title])
    h = 0.62 + 0.205 * len(lines)
    fig = plt.figure(figsize=(max(7.0, 0.085 * width + 0.6), h), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, h)
    ax.add_patch(plt.Rectangle((0, h - 0.36), 1, 0.36, color=BAR))
    step = 0.16 / fig.get_figwidth()
    ax.scatter([0.018 + k * step for k in range(3)], [h - 0.18] * 3, s=18,
               c=["#ff5f57", "#febc2e", "#28c840"], zorder=3)
    ax.text(0.5, h - 0.18, title, color=DIM, ha="center", va="center", family="monospace", fontsize=7.5)
    y = h - 0.56
    for l in lines:
        col = DIM if l == "..." else (HI if any(l.lstrip().startswith(x) or x in l for x in hl) else FG)
        ax.text(0.015, y, l, color=col, family="monospace", fontsize=7.2, va="center")
        y -= 0.205
    fig.savefig(os.path.join(OUT, fname), facecolor=BG); plt.close(fig)


def main():
    render(excerpt(read("train_eval_sentinel_official.log"), 6, 9, keep=("lr ->",)),
           "train_eval_sentinel_official.log  (official split)", "term_train_official.png")
    render(excerpt(read("train_eval_sentinel_groupsplit.log"), 6, 9, keep=("lr ->",)),
           "train_eval_sentinel_groupsplit.log  (leak-free split)", "term_train_groupsplit.png")
    render([l for l in read("eval_peru.log") if not l[:1].isdigit()], "eval_peru.log  (both models)",
           "term_peru.png")
    # MDOD: the log file is interleaved with a killed earlier run, so show the final summary tables,
    # which are the clean console output of the finished job.
    L, keep = read("eval_mdod.log"), []
    for i, l in enumerate(L):
        if l.startswith("=="):
            keep.append(l)
        if l.startswith("class "):
            keep += L[i:i + 7]
    render(keep, "evaluate_mdod.py  (summary, both models)", "term_mdod.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
