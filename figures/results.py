import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALL_PROSODIC, FIGURE_DIR, PROSODY_DIR, TABLE_DIR

PT = str(TABLE_DIR)
OUT = str(FIGURE_DIR)
RED, BLUE = "#e74c3c", "#3498db"
ORANGE, GREY = "#f39c12", "#95a5a6"
LANGS = ["Dutch", "English", "German", "Mandarin"]
CHANCE = 0.500


def save(fig, name, dpi=300):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("written", p)


COL_W, PRINT_FS = 3.45, 7
SOFT_RED, SOFT_BLUE, INK, GRID = "#d6604d", "#4393c3", "0.2", "0.88"
PRINT_STYLE = {"font.size": PRINT_FS, "axes.linewidth": 0.5, "axes.edgecolor": INK,
               "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
               "xtick.major.width": 0.5, "ytick.major.width": 0.5,
               "xtick.major.size": 2, "ytick.major.size": 0, "xtick.major.pad": 1.5,
               "ytick.major.pad": 2, "axes.titlesize": PRINT_FS + 0.5,
               "axes.titleweight": "bold", "axes.titlepad": 3,
               "legend.fontsize": PRINT_FS - 0.5, "legend.frameon": False,
               "legend.handlelength": 1.0, "legend.handletextpad": 0.4}


def clean(ax, grid="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def holm(p):
    o = np.argsort(p)
    m, a, prev = len(p), np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        prev = max(prev, min(1.0, (m - r) * p[i]))
        a[i] = prev
    return a


def best_budget():
    r = pd.read_csv(os.path.join(PT, "monolingual.csv"))
    return r.rename(columns={"nested_mean": "mean_f1", "nested_sd": "sd",
                             "nested_min": "min_f1", "nested_max": "max_f1"})


def figure_D():
    from matplotlib.lines import Line2D
    b = best_budget()
    mean = b.pivot_table(index="tier", columns="corpus", values="mean_f1")
    sd = b.pivot_table(index="tier", columns="corpus", values="sd")
    order = mean.mean(axis=1).sort_values(ascending=False).index.tolist()
    emo = [t.endswith("MSP") or t == "emotion2vec" for t in order]
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(COL_W, 1.62), sharey=True)
        yy = np.arange(len(order))[::-1]
        for ax, lang in zip(axes, LANGS):
            ax.axvline(CHANCE, color="0.45", ls=(0, (3, 2)), lw=0.6)
            for yi, t, e in zip(yy, order, emo):
                c = SOFT_RED if e else SOFT_BLUE
                ax.errorbar(mean.loc[t, lang], yi, xerr=sd.loc[t, lang], fmt="o", ms=3,
                            color=c, ecolor=c, elinewidth=0.8, capsize=0, mew=0)
            ax.set_xlim(0.45, 0.86)
            ax.set_xticks([0.5, 0.6, 0.7, 0.8])
            ax.set_xticklabels(["0.5", "", "0.7", ""])
            ax.set_title(lang)
            clean(ax)
        axes[0].set_yticks(yy)
        axes[0].set_yticklabels(order)
        axes[0].set_ylim(-0.6, len(order) - 0.4)
        fig.text(0.62, 0.075, "Macro-F1, mean $\\pm$ sd over twenty draws", ha="center",
                 fontsize=PRINT_FS, color=INK)
        dot = lambda c, l: Line2D([], [], color=c, marker="o", ms=3, lw=0.8, label=l)
        fig.legend(handles=[dot(SOFT_RED, "trained on emotional speech"),
                            dot(SOFT_BLUE, "other representations")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.08), columnspacing=1.2)
        plt.tight_layout(pad=0.2, w_pad=0.5, rect=(0, 0.13, 1, 1))
        save(fig, "fig_monolingual_spread.png")


def figure_E():
    s = pd.read_csv(os.path.join(PT, "contrastive_ablation.csv"))
    b = s.rename(columns={"wilcoxon_p": "wilcoxon_p_if_best"}).copy()
    order_c = ["Dutch", "English", "German", "Mandarin"]
    order_t = ["7 Core Prosodic", "emotion2vec", "Wav2Vec2-Large MSP", "WavLM-Large MSP"]
    b["ci"] = pd.Categorical(b.corpus, order_c, ordered=True)
    b["ti"] = pd.Categorical(b.tier, order_t, ordered=True)
    b = b.sort_values(["ci", "ti"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(b))[::-1]
    for i, r in b.iterrows():
        col = RED if r.delta_pp > 0 else BLUE
        sig = r.wilcoxon_p_if_best < 0.05
        ax.plot([0, r.delta_pp], [y[i], y[i]], color=col, lw=2.2, zorder=2)
        ax.plot(r.delta_pp, y[i], "o", ms=11, zorder=3, color=col if sig else "white",
                markeredgecolor=col, markeredgewidth=2.0)
        ax.text(r.delta_pp + (0.18 if r.delta_pp >= 0 else -0.18), y[i],
                "%+.2f%s" % (r.delta_pp, " *" if sig else ""),
                va="center", ha="left" if r.delta_pp >= 0 else "right",
                fontsize=10, fontweight="bold" if sig else "normal")
    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(["%s  --  %s" % (r.corpus, r.tier) for _, r in b.iterrows()],
                       fontsize=10)
    for k in range(1, 4):
        ax.axhline(y[4 * k] + 0.5, color="0.75", lw=0.8, ls=":")
    lo, hi = b.delta_pp.min(), b.delta_pp.max()
    ax.set_xlim(lo - 0.20 * (hi - lo) - 0.9, hi + 0.20 * (hi - lo) + 0.9)
    ax.set_xlabel("Macro-F1 gained by adding the SupCon head (pp)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    save(fig, "fig_supcon_ablation.png")


def figure_F():
    order = ["Dutch", "English", "Mandarin", "German"]
    d = pd.read_csv(os.path.join(PT, "joint_vs_specialists.csv"))
    ct = pd.read_csv(os.path.join(PT, "corrected_tests.csv"))
    draws = int(d.n_draws.max())
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.7), sharey=True)
        for ax, tier, nice in [(axes[0], "WavLM-Large MSP", "WavLM-Large MSP (1,024-dim, learned)"),
                               (axes[1], "7 Core Prosodic", "7 Core Prosodic (7-dim, handcrafted)")]:
            t = d[d.tier == tier].set_index("corpus").reindex(order).reset_index()
            c = ct[ct.comparison == "joint - specialist, %s" % tier].set_index("corpus")
            t["holm"] = t.corpus.map(c.p_corrected_holm)
            x, w = np.arange(len(order)), 0.34
            err = dict(elinewidth=0.6, ecolor="0.35", capsize=0)
            ax.bar(x - w / 2 - 0.01, t.mono_mean, w, yerr=t.mono_sd, color=SOFT_BLUE,
                   label="per-language specialist", error_kw=err)
            ax.bar(x + w / 2 + 0.01, t.joint_mean, w, yerr=t.joint_sd, color=SOFT_RED,
                   label="one joint model, four languages", error_kw=err)
            for i, r in t.iterrows():
                top = max(r.mono_mean + r.mono_sd, r.joint_mean + r.joint_sd)
                sig = (r.holm == r.holm) and r.holm < 0.05
                ax.annotate("%+.1f pp%s" % (r.delta_pp, "*" if sig else ""), xy=(i, top + 0.018),
                            ha="center", fontsize=PRINT_FS, color=SOFT_RED if sig else "0.35",
                            fontweight="bold" if sig else "normal")
            ax.axhline(CHANCE, color="0.45", ls=(0, (3, 2)), lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(order)
            ax.tick_params(axis="x", length=0)
            ax.set_title(nice)
            ax.set_ylim(0.35, 0.92)
            ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            ax.tick_params(axis="y", length=2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Macro-F1 (mean $\\pm$ sd,\n%d paired draws)" % draws)
        axes[1].legend(loc="upper right", borderaxespad=0.1, labelspacing=0.3)
        plt.tight_layout(pad=0.2, w_pad=1.2)
        save(fig, "fig_joint_vs_monolingual.png")


def figure_G():
    from matplotlib.patches import Patch
    STABLE = ["duration_ms", "pitch_slope", "energy_slope"]
    NICE = {"duration_ms": "Duration", "pitch_slope": "Pitch slope",
            "energy_slope": "Energy slope", "intensity_mean_db": "Mean intensity",
            "pitch_mean_hz": "Mean pitch", "hnr_db": "HNR", "shimmer_pct": "Shimmer",
            "jitter_pct": "Jitter", "pitch_max_hz": "Pitch maximum",
            "pitch_min_hz": "Pitch minimum", "pitch_span_st": "Pitch span",
            "pitch_std_hz": "Pitch sd", "spectral_centroid_hz": "Spectral centroid",
            "pause_ratio": "Pause ratio"}

    df = pd.read_csv(PROSODY_DIR / "nl_prosody.csv").dropna(subset=ALL_PROSODIC, how="all")
    y = df.label.str.lower() == "sarcastic"

    def cd(c):
        a, b_ = df.loc[y, c].dropna(), df.loc[~y, c].dropna()
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1))
                     / (len(a) + len(b_) - 2))
        return abs((a.mean() - b_.mean()) / sp) if sp else 0.0

    d = pd.Series({c: cd(c) for c in ALL_PROSODIC}).sort_values(ascending=False)

    L = pd.read_csv(os.path.join(PT, "feature_importance.csv")).set_index("feature").reindex(d.index)
    with plt.rc_context(PRINT_STYLE):
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(COL_W, 1.95), sharey=True)
        yy = np.arange(len(d))[::-1]
        cols = [SOFT_RED if f in STABLE else SOFT_BLUE for f in d.index]
        a0.barh(yy, d.values, color=cols, height=0.62)
        a0.axvline(0.2, color="0.45", ls=(0, (3, 2)), lw=0.6)
        a0.set_xlim(0, 0.66)
        a0.set_xticks([0, 0.2, 0.4, 0.6])
        a0.set_xlabel("|Cohen's $d|$")
        a0.set_title("Effect size")
        a1.barh(yy, L.mean_importance.values, xerr=L.std_importance.values, color=cols,
                height=0.62, error_kw=dict(elinewidth=0.5, ecolor="0.45", capsize=0))
        a1.axvline(0, color=INK, lw=0.5)
        a1.set_xticks([0, 0.05, 0.1])
        a1.set_xlabel("Permutation importance")
        a1.set_title("LOSO permutation")
        a0.set_yticks(yy)
        a0.set_yticklabels([NICE[f] for f in d.index])
        a0.set_ylim(-0.6, len(d) - 0.4)
        for ax in (a0, a1):
            clean(ax)
        fig.legend(handles=[Patch(color=SOFT_RED, label="sign-stable across all four languages"),
                            Patch(color=SOFT_BLUE, label="other candidate")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.01), columnspacing=1.2)
        plt.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0.07, 1, 1))
        save(fig, "fig_dutch_importance.png")


def figure_H():
    from matplotlib.ticker import MaxNLocator
    rc = PT
    c = pd.read_csv(os.path.join(rc, "layerwise_curve.csv"))
    draws = int(pd.read_csv(os.path.join(rc, "layerwise_summary.csv")).n_draws.max())
    ch = pd.read_csv(os.path.join(rc, "layerwise_pipeline_draws.csv"))
    encs = [("XLSR-53", "XLSR-53"), ("Wav2Vec2-Large MSP", "Wav2Vec2-Large MSP"),
            ("WavLM-Large MSP", "WavLM-Large MSP"), ("emotion2vec", "emotion2vec_base")]
    order = ["Dutch", "English", "Mandarin", "German"]
    col = dict(zip(order, [BLUE, ORANGE, "#2ecc71", RED]))
    row = dict(zip(order, [0.445, 0.425, 0.405, 0.385]))
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
    for ax, (enc, nice) in zip(axes, encs):
        for corpus in order:
            t = c[(c.encoder == enc) & (c.corpus == corpus)].sort_values("layer")
            ax.plot(t.layer, t.mean_f1, marker="o", ms=3, lw=1.4, color=col[corpus],
                    label=corpus)
            k = ch[(ch.encoder == enc) & (ch.corpus == corpus)].layer
            ax.plot([k.min(), k.max()], [row[corpus]] * 2, lw=1.6, color=col[corpus],
                    solid_capstyle="butt")
            ax.plot([k.median()], [row[corpus]], marker="o", ms=4.5, color=col[corpus],
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.axhline(CHANCE, color="0.4", ls="--", lw=1.0)
        ax.axhline(0.465, color="0.8", lw=0.6)
        ax.set_title(nice, fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylim(0.37, 0.81)
    axes[0].set_yticks([0.5, 0.6, 0.7, 0.8])
    axes[0].set_ylabel("Macro-F1 (mean, %d draws)" % draws, fontsize=11)
    axes[0].text(-0.5, 0.47, "chosen layer", fontsize=8.5, color="0.35", va="bottom")
    axes[-1].legend(fontsize=8.5, loc="lower center", ncol=2, frameon=False,
                    bbox_to_anchor=(0.5, 0.30), borderaxespad=0, labelspacing=0.25,
                    columnspacing=1.0, handlelength=1.3)
    plt.tight_layout()
    save(fig, "fig_layerwise_curve.png")


FIGS = {"D": figure_D, "E": figure_E, "F": figure_F, "G": figure_G, "H": figure_H}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=list(FIGS) + ["all"], default="all")
    a = ap.parse_args()
    for k in (list(FIGS) if a.figure == "all" else [a.figure]):
        FIGS[k]()
