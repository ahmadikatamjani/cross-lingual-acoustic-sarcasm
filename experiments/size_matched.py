import glob
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R

HF = Path(os.environ.get("HF_HUB_CACHE",
                         Path.home() / ".cache" / "huggingface" / "hub"))
CHECKPOINT = {"XLSR-53": "facebook/wav2vec2-large-xlsr-53",
              "WavLM-Large MSP": "tiantiaf/wavlm-large-msp-podcast-emotion-dim",
              "Wav2Vec2-Large MSP": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"}
PAIRS = [("WavLM-Large MSP", "XLSR-53"), ("Wav2Vec2-Large MSP", "XLSR-53")]
OUT = R.OUT / "size_matched.csv"


def size(tier):
    from safetensors import safe_open
    repo = "models--" + CHECKPOINT[tier].replace("/", "--")
    hits = glob.glob(str(HF / repo / "snapshots" / "*" / "model.safetensors"))
    if not hits:
        raise SystemExit("%s: no model.safetensors under %s" % (tier, HF / repo))
    with safe_open(hits[0], "pt") as f:
        keys = list(f.keys())
        n = sum(math.prod(f.get_slice(k).get_shape()) for k in keys)
    layers = {int(m.group(1)) for k in keys
              for m in [re.search(r"layers\.(\d+)\.", k)] if m}
    return len(layers), n / 1e6


def same_draws(tiers, code):
    sig = []
    for tier in tiers:
        spec = [t for t in R.TIERS if t[0] == tier][0]
        _, y, spk = R.load(*spec, code)
        spk = np.asarray(spk).astype(str)
        u = np.unique(spk)
        sig.append((tuple(u), tuple(int((spk == s).sum()) for s in u),
                    tuple(int(y[spk == s].sum()) for s in u)))
    return all(s == sig[0] for s in sig[1:])


def holm(p):
    o = np.argsort(p)
    m, out, prev = len(p), np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        prev = max(prev, min(1.0, (m - r) * p[i]))
        out[i] = prev
    return out


def main():
    wide = pd.read_csv(R.OUT / "monolingual.csv")

    def scores(tier, corpus):
        r = wide[(wide.tier == tier) & (wide.corpus == corpus)]
        return np.array([float(v) for v in r.scores.iloc[0].split(";")])

    sizes = {t: size(t) for t in CHECKPOINT}
    for t, (L, n) in sizes.items():
        print("%-20s %2d layers  %6.1f M   (%s)" % (t, L, n, CHECKPOINT[t]))

    rows = []
    for a, b in PAIRS:
        block = []
        for code, corpus in R.CORPORA.items():
            if not same_draws([a, b], code):
                raise SystemExit("%s vs %s on %s: different speakers" % (a, b, corpus))
            A, B = scores(a, corpus), scores(b, corpus)
            if len(A) != len(B):
                raise SystemExit("%s: %d draws against %d" % (corpus, len(A), len(B)))
            d = A - B
            block.append(dict(pair="%s vs %s" % (a, b), tier_a=a, tier_b=b,
                              layers_a=sizes[a][0], layers_b=sizes[b][0],
                              params_a_m=round(sizes[a][1], 1),
                              params_b_m=round(sizes[b][1], 1),
                              size_matched=sizes[a][0] == sizes[b][0]
                              and abs(sizes[a][1] - sizes[b][1]) / sizes[b][1] < 0.02,
                              corpus=corpus, n_draws=len(d),
                              mean_a=round(A.mean(), 6), mean_b=round(B.mean(), 6),
                              delta_pp=round(d.mean() * 100, 4),
                              p=float(wilcoxon(A, B).pvalue),
                              a_better_draws=int((d > 0).sum()),
                              same_draws=True))
        hp = holm(np.array([r["p"] for r in block]))
        for r, h in zip(block, hp):
            r["holm_p"] = float(h)
        rows += block

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    for r in out.itertuples():
        print("%-40s %-9s %+6.2f pp  Holm p = %.4g" % (r.pair, r.corpus, r.delta_pp, r.holm_p))


if __name__ == "__main__":
    main()
