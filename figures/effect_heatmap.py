import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import FIGURE_DIR, TABLE_DIR
import style as figstyle

OUT = FIGURE_DIR / "quad_lingual_heatmap.png"
LANGS = ["Dutch", "English", "German", "Mandarin"]

NAMES = [
    ("duration_ms", "Duration"),
    ("energy_slope", "Energy slope"),
    ("pitch_slope", "Pitch slope"),
    ("hnr_db", "HNR"),
    ("pitch_mean_hz", "Mean pitch"),
    ("intensity_mean_db", "Mean intensity"),
    ("shimmer_pct", "Shimmer"),
]


def main():
    figstyle.apply()
    d = (pd.read_csv(TABLE_DIR / "effect_sizes_core7.csv")
         .rename(columns={"Unnamed: 0": "feature"}).set_index("feature"))
    M = np.array([[d.loc[k, L] for L in LANGS] for k, _ in NAMES], dtype=float)

    fig, ax = plt.subplots(figsize=(figstyle.COL, 2.9))
    lim = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")

    ax.set_xticks(range(len(LANGS)), LANGS)
    ax.set_yticks(range(len(NAMES)), [n for _, n in NAMES])
    ax.set_xticks(np.arange(-.5, len(LANGS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "%+.2f" % v, ha="center", va="center",
                    fontsize=figstyle.ANNOT,
                    color="white" if abs(v) > 0.62 * lim else "black")

    ax.axhline(2.5, color="#2c3e50", linewidth=1.1)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Cohen's $d$", fontsize=figstyle.LABEL)
    cb.ax.tick_params(labelsize=figstyle.ANNOT)
    cb.outline.set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, OUT)
    print("written %s" % OUT)


if __name__ == "__main__":
    main()
