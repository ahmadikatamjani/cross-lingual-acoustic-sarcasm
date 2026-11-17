import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALL_PROSODIC, DUTCH_FEATURE_DIR, PROSODY_DIR

MFCC = ["mfcc_%02d" % (i + 1) for i in range(13)]
MFCC_SD = ["mfccsd_%02d" % (i + 1) for i in range(13)]


def load(code):
    df = pd.read_csv(PROSODY_DIR / ("%s_prosody.csv" % code))
    df = df.dropna(subset=ALL_PROSODIC, how="all").reset_index(drop=True)
    mf = pd.read_csv(PROSODY_DIR / ("%s_mfcc.csv" % code))
    keep = [c for c in MFCC + MFCC_SD if c in mf.columns]
    return df.merge(mf[keep + ["utt_id"]], on="utt_id", how="left")


def dutch_40():
    df = load("nl")
    out = df[ALL_PROSODIC].copy()
    for i in range(13):
        m, s = "mfcc_%02d" % (i + 1), "mfccsd_%02d" % (i + 1)
        out["mfcc%02d_mean" % i] = df[m]
        out["mfcc%02d_std" % i] = df[s]
    out["label"] = df["label"]
    out["participant_id"] = df["speaker_id"]
    DUTCH_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DUTCH_FEATURE_DIR / "dutch_40_features.csv", index=False)
    print("dutch_40_features.csv: %d rows, %d features" % (len(out), len(ALL_PROSODIC) + 26))


if __name__ == "__main__":
    dutch_40()
