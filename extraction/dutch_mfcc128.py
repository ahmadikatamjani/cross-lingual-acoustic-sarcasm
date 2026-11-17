import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DUTCH_AUDIO, DUTCH_FEATURE_DIR, DUTCH_METADATA


def extract_mfccs(audio_path, n_mfcc=128):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=128), axis=1)
    except Exception as e:
        print("%s: %s" % (audio_path, e))
        return None


def main():
    label_map = dict(zip(*pd.read_csv(DUTCH_METADATA)[["utterance_file", "label"]].values.T))
    records = []
    for f in tqdm(sorted(DUTCH_AUDIO.glob("*.wav"))):
        if f.name not in label_map:
            continue
        mfcc_128 = extract_mfccs(str(f))
        if mfcc_128 is not None:
            row = {"filename": f.name, "label": label_map[f.name]}
            row.update({"mfcc_%03d" % i: v for i, v in enumerate(mfcc_128)})
            records.append(row)
    DUTCH_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(DUTCH_FEATURE_DIR / "dutch_128_mfcc_features.csv", index=False)
    print("%d files" % len(records))


if __name__ == "__main__":
    main()
