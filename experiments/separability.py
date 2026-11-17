import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DUTCH_FEATURE_DIR, FEATURE_DIR, PAPER_7, PROSODY_DIR, TABLE_DIR

LANGS = {"nl": "Dutch", "en": "English", "de": "German", "zh": "Mandarin"}
CORPORA = ["Dutch", "English", "German", "Mandarin"]
STEM = {"nl": "dutch", "en": "mustardpp", "de": "musag", "zh": "mcsd"}
TIERS = [
    ("7 Core Prosodic", None, None),
    ("eGeMAPS", "egemaps", "egemaps"),
    ("ComParE 2016", "compare", "compare"),
    ("XLSR-53", "xlsr", "xlsr"),
    ("emotion2vec_base", "emotion2vec", "emotion2vec"),
    ("Wav2Vec2-Large MSP", "wav2vec2_msp", "wav2vec2_msp"),
    ("WavLM-Large MSP", "wavlm_msp", "wavlm_msp"),
]
DUTCH_CSV = {"egemaps": "dutch_egemaps_features.csv", "compare": "dutch_compare_features.csv"}
META = {"label", "participant_id", "filename", "id", "name", "file", "speaker_id"}


def labels_from(raw):
    if raw.dtype.kind in ("U", "S", "O"):
        return np.array([1 if str(v).strip() == "sarcastic" else 0 for v in raw])
    return raw.astype(int)


def numeric(df):
    y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
    cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols], y


def load(tier, subdir, frag, code):
    if tier == "7 Core Prosodic":
        p = PROSODY_DIR / ("%s_prosody.csv" % code)
        if not p.exists():
            return None
        df = pd.read_csv(p).dropna(subset=PAPER_7)
        y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
        return df[PAPER_7].values.astype(float), y
    if code == "nl" and frag in DUTCH_CSV:
        p = DUTCH_FEATURE_DIR / DUTCH_CSV[frag]
        if not p.exists():
            return None
        X, y = numeric(pd.read_csv(p).dropna())
        return X.values.astype(float), y
    if subdir == "xlsr":
        hits = [p for p in [FEATURE_DIR / "xlsr" / ("%s_xlsr_embeddings.npz" % code)] if p.exists()]
    else:
        hits = list((FEATURE_DIR / subdir).glob("%s*%s*.npz" % (STEM[code], frag)))
    if not hits:
        return None
    d = np.load(hits[0], allow_pickle=True)
    if "labels" not in d.files:
        return None
    return d["embeddings"].astype(np.float64), labels_from(d["labels"])


def sil_table(path):
    if not Path(path).exists():
        return None
    X, y = numeric(pd.read_csv(path))
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    keep = ~X.isna().any(axis=1)
    X, y = X[keep], y[keep.values]
    if len(np.unique(y)) < 2:
        return None
    return float(silhouette_score(StandardScaler().fit_transform(X), y))


def tier_row(tier, subdir, frag):
    rec = {"feature_set": tier}
    for code in ["nl", "en", "de", "zh"]:
        got = load(tier, subdir, frag, code)
        if got is None or len(np.unique(got[1])) < 2 or len(got[1]) < 10:
            rec[LANGS[code]] = None
            continue
        X, y = got
        rec[LANGS[code]] = round(float(silhouette_score(StandardScaler().fit_transform(X), y)), 4)
    return rec


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    base = {t: tier_row(t, s, f) for t, s, f in TIERS}
    mfcc26 = {LANGS[c]: sil_table(PROSODY_DIR / ("%s_mfcc.csv" % c)) for c in LANGS}
    rows = [
        dict(base["7 Core Prosodic"], dim="7"),
        dict(feature_set="MFCC-26", dim="26", **mfcc26),
        dict(feature_set="MFCC-128", dim="128",
             Dutch=sil_table(DUTCH_FEATURE_DIR / "dutch_128_mfcc_features.csv")),
        dict(feature_set="40 Prosodic + MFCC", dim="40",
             Dutch=sil_table(DUTCH_FEATURE_DIR / "dutch_40_features.csv")),
        dict(base["eGeMAPS"], dim="88"),
        dict(base["ComParE 2016"], dim="6373"),
        dict(base["XLSR-53"], dim="1024"),
        dict(base["emotion2vec_base"], dim="768"),
        dict(base["Wav2Vec2-Large MSP"], dim="1024"),
        dict(base["WavLM-Large MSP"], dim="1024"),
    ]
    df = pd.DataFrame(rows)[["feature_set", "dim"] + CORPORA]
    df.to_csv(TABLE_DIR / "silhouette.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
