import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = 344.0 / 72.0
BODY_PT = 10.0

TITLE = 9
LABEL = 8.5
TICK = 8
ANNOT = 7
LEGEND = 7.5


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Open Sans", "DejaVu Sans"],
        "font.size": TICK,
        "axes.titlesize": TITLE,
        "axes.labelsize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        "figure.titlesize": TITLE,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, path, **kw):
    path = str(path)
    kw.setdefault("bbox_inches", "tight")
    kw.setdefault("pad_inches", 0.02)
    fig.savefig(path.replace(".png", ".pdf"), **kw)
    fig.savefig(path, **kw)
    return path
