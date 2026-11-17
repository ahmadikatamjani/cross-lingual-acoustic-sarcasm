import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TABLE_DIR

OUT = TABLE_DIR
BLOCKS = ["Wav2Vec2-Large MSP", "WavLM-Large MSP", "7 Core Prosodic"]
SRC = {"NL": "Dutch", "EN": "English", "ZH": "Chinese", "DE": "German"}
CORPUS = {"NL": "Dutch", "EN": "English", "ZH": "Mandarin", "DE": "German"}


def zero_shot(arm="frozen"):
    d = pd.read_csv(OUT / "zeroshot_standardised.csv")
    d = d[d.arm == arm]
    code = {v: k for k, v in SRC.items()}
    out = {}
    for block in BLOCKS:
        g = d[d.block == block]
        if len(g) != 12:
            raise SystemExit("%s: %d cells, expected 12" % (block, len(g)))
        out[block] = {}
        for r in g.itertuples():
            out[block].setdefault(r.source, {})[code[r.target]] = float(r.mean_f1)
    return out


def monolingual(features):
    d = pd.read_csv(OUT / "monolingual.csv")
    out = {}
    for feat in features:
        out[feat] = {}
        for code, corpus in CORPUS.items():
            row = d[(d.tier == feat) & (d.corpus.str.startswith(corpus[:4]))]
            if len(row) != 1:
                raise SystemExit("%s/%s matched %d rows" % (feat, corpus, len(row)))
            sc = [float(v) for v in str(row.scores.iloc[0]).split(";") if v.strip()]
            out[feat][code] = float(np.mean(sc))
    return out


def tax_table(zero, mono):
    rows = []
    for feat, block in zero.items():
        for tgt in ("NL", "EN", "ZH", "DE"):
            cells = {s: v[tgt] for s, v in block.items() if tgt in v}
            best_src = max(cells, key=cells.get)
            best = cells[best_src]
            m = mono[feat][tgt]
            rows.append(dict(feature_set=feat, target=tgt, best_source=best_src,
                             best_zeroshot=round(best, 6),
                             monolingual=round(m, 6),
                             tax_pp=round((m - best) * 100, 4),
                             mean_zeroshot=round(float(np.mean(list(cells.values()))), 6)))
    return pd.DataFrame(rows)


def main():
    frozen = zero_shot("frozen")
    mono = monolingual(frozen)
    tgt = tax_table(zero_shot("target"), mono)
    tgt.to_csv(OUT / "language_tax_target.csv", index=False)
    df = tax_table(frozen, mono)
    df.to_csv(OUT / "language_tax.csv", index=False)
    print(df.to_string(index=False))
    print("language tax %.1f to %.1f pp, mean %.1f; standardised on the target, mean %.1f"
          % (df.tax_pp.min(), df.tax_pp.max(), df.tax_pp.mean(), tgt.tax_pp.mean()))


if __name__ == "__main__":
    main()
