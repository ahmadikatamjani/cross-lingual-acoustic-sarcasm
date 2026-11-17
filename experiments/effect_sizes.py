import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALL_PROSODIC, PAPER_7, PROSODY_DIR, TABLE_DIR

LANGS = [("nl", "Dutch"), ("en", "English"), ("de", "German"), ("zh", "Mandarin")]
MFCC = ["mfcc_%02d" % (i + 1) for i in range(13)]
SETS = {"effect_sizes_core7": (PAPER_7, False),
        "effect_sizes_all14": (ALL_PROSODIC, False),
        "effect_sizes_with_mfcc": (ALL_PROSODIC + MFCC, True)}


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp else np.nan


def build(feats, with_mfcc):
    rows = {}
    for code, name in LANGS:
        df = pd.read_csv(PROSODY_DIR / ("%s_prosody.csv" % code))
        if with_mfcc:
            mf = pd.read_csv(PROSODY_DIR / ("%s_mfcc.csv" % code))
            df = df.merge(mf[MFCC + ["utt_id"]], on="utt_id", how="left")
        y = df["label"].astype(str).str.lower() == "sarcastic"
        rows[name] = {c: cohens_d(df.loc[y, c].dropna(), df.loc[~y, c].dropna())
                      for c in feats if c in df.columns}
    return pd.DataFrame(rows)[[n for _, n in LANGS]]


def order_rows(m):
    cons = [f for f in m.index if m.loc[f].notna().all()
            and len(set(np.sign(m.loc[f].dropna()))) == 1]
    rev = [f for f in m.index if f not in cons]
    cons.sort(key=lambda f: -abs(m.loc[f]).min())
    rev.sort(key=lambda f: -abs(m.loc[f]).max())
    return cons, rev


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for stem, (feats, with_mfcc) in SETS.items():
        m = build(feats, with_mfcc).dropna(how="all")
        cons, rev = order_rows(m)
        m = m.loc[cons + rev]
        m.round(4).to_csv(TABLE_DIR / (stem + ".csv"))
        print("%s: same sign in every language: %s" % (stem, ", ".join(cons)))


if __name__ == "__main__":
    main()
