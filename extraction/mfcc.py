import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prosody import CORPORA, OUT_DIR, SOURCES, SR, mp4_to_wav

N_MFCC = 13
COLS = ["mfcc_%02d" % (i + 1) for i in range(N_MFCC)]
SD_COLS = ["mfccsd_%02d" % (i + 1) for i in range(N_MFCC)]


def run(code, limit=None):
    name, gen = CORPORA[code]
    items = list(gen())
    if limit:
        items = items[:limit]
    rows, missing, failed = [], 0, 0
    for path, label, spk, uid in tqdm(items, desc=name):
        if not path.exists():
            missing += 1
            continue
        tmp = None
        try:
            if path.suffix.lower() in (".mp4", ".mkv", ".avi"):
                tmp = mp4_to_wav(path)
                y, _ = librosa.load(str(tmp), sr=SR, mono=True)
            else:
                y, _ = librosa.load(str(path), sr=SR, mono=True)
            if y.size < SR // 20 or float(np.max(np.abs(y))) <= 0.0:
                failed += 1
                continue
            M = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
            r = {c: float(v) for c, v in zip(COLS, M.mean(axis=1))}
            r.update({c: float(v) for c, v in zip(SD_COLS, M.std(axis=1))})
            r.update(label=label, speaker_id=spk, utt_id=uid)
            rows.append(r)
        except Exception as e:
            failed += 1
            print("%s: %s" % (path.name, e))
        finally:
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("%s_mfcc.csv" % code)
    df = pd.DataFrame(rows)[COLS + SD_COLS + ["label", "speaker_id", "utt_id"]]
    df.to_csv(out, index=False)
    print("%s: rows=%d  missing=%d  failed=%d" % (name, len(df), missing, failed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for c in (list(CORPORA) if a.corpus == "all" else [a.corpus]):
        if not SOURCES[c].exists():
            print("%s: %s not found, skipped" % (c, SOURCES[c]))
            continue
        run(c, a.limit)
