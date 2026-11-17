import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import FIGURE_DIR
import style as figstyle

OUT = FIGURE_DIR / "pipeline_schematic.png"

FIG_W, FIG_H = 516.0 / 72.0, 3.3

BODY = 8.0
HEAD = 7.5
LW = 0.78
LINE = 4.2
EDGE, ARROW_C, MUTED = "#2c3e50", "#34495e", "#5d6d7e"


def box(ax, x, y, w, h, title, body, bg):
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=LW, edgecolor=EDGE,
                                   facecolor=bg, zorder=2))
    lines = body.count("\n") + 1 if body else 0
    top = y + h / 2 + (lines + 1) * LINE / 2
    ax.text(x + w / 2, top, title, fontsize=BODY, fontweight="bold",
            ha="center", va="top", zorder=3)
    if body:
        ax.text(x + w / 2, top - LINE, body, fontsize=BODY, ha="center",
                va="top", zorder=3, linespacing=1.15)
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y)}


def arrow(ax, start, end, rad=0.0):
    ax.annotate("", xy=end, xycoords="data", xytext=start, textcoords="data",
                zorder=1,
                arrowprops=dict(arrowstyle="->", color=ARROW_C, lw=LW,
                                mutation_scale=7, shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=%s" % rad))


def line(ax, xs, ys):
    ax.plot(xs, ys, color=ARROW_C, lw=LW, solid_capstyle="butt", zorder=1)


def bus(ax, x, ys):
    line(ax, [x, x], [min(ys), max(ys)])


def head(ax, x, text):
    ax.text(x, 99, text, fontsize=HEAD, style="italic", color=MUTED,
            ha="center", va="top")


def main(out):
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    head(ax, 9, "Corpora")
    c = box(ax, 0.5, 32, 17, 36, "Four corpora",
            "Dutch (NL)  1,771\nEnglish (EN)  1,201\nGerman (DE)  214\n"
            "Mandarin (ZH)  2,535\n(utterances)", "#ecf0f1")

    head(ax, 34.5, "Representations (dimension)")
    t1 = box(ax, 21, 66, 27, 27, "Tier 1: handcrafted",
             "7 Core Prosodic (7)\nMFCC-26 (26), MFCC-128 (128)\n"
             "Prosodic + MFCC (40)\nMFCC sets: Dutch only", "#d5f5e3")
    t2 = box(ax, 21, 42, 27, 19, "Tier 2: standard sets",
             "eGeMAPS (88)\nComParE 2016 (6,373)", "#d6eaf8")
    t3 = box(ax, 21, 4, 27, 33, "Tier 3: pre-trained encoders",
             "mean-pooled per utterance\nXLSR-53 (1,024)\n"
             "Wav2Vec2-Large MSP (1,024)\nWavLM-Large MSP (1,024)\n"
             "emotion2vec (768)", "#f5eef8")
    arrow(ax, c["r"], t1["l"], rad=-0.15)
    arrow(ax, c["r"], t2["l"])
    arrow(ax, c["r"], t3["l"], rad=0.15)

    head(ax, 60.5, "Shared model")
    m1 = box(ax, 52, 68.5, 17, 22, "Standardise",
             "fitted on the\ntraining data", "#fcf3cf")
    m2 = box(ax, 52, 39, 17, 22, "SupCon encoder",
             "d → 256 → 64\nL2-normalised", "#fcf3cf")
    m3 = box(ax, 52, 10, 17, 22, "RBF-SVM", "C = 1,\nbalanced classes",
             "#fcf3cf")
    xb = 50.0
    for t in (t1, t2, t3):
        line(ax, [t["r"][0], xb], [t["r"][1]] * 2)
    bus(ax, xb, [t1["r"][1], t3["r"][1]])
    arrow(ax, (xb, m1["l"][1]), m1["l"])
    arrow(ax, m1["b"], m2["t"])
    arrow(ax, m2["b"], m3["t"])

    head(ax, 86.5, "Three separate paradigms")
    p1 = box(ax, 74, 67, 25, 26, "Monolingual",
             "train: one language\ntest: its unseen speakers\n"
             "20 speaker draws", "#fadbd8")
    p2 = box(ax, 74, 37, 25, 26, "Zero-shot transfer",
             "train: one language\ntest: each other language\n"
             "12 pairs, 20 seeds\nscaler: source or target audio", "#fadbd8")
    p3 = box(ax, 74, 4, 25, 29, "Joint multilingual",
             "train: all four pooled\ntest: unseen speakers of each\n"
             "20 draws; also\nleave-one-language-out", "#fadbd8")
    xp = 71.5
    line(ax, [m3["r"][0], xp], [m3["r"][1]] * 2)
    bus(ax, xp, [p1["l"][1], p3["l"][1]])
    for p in (p1, p2, p3):
        arrow(ax, (xp, p["l"][1]), p["l"])

    out.parent.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, out)
    print("written %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    main(Path(ap.parse_args().out))
