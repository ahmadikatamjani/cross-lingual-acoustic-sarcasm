import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALL_PROSODIC as FEATURE_COLS
from config import (DUTCH_AUDIO, DUTCH_METADATA, MCSD_DIR, MUSAG_DIR, MUSTARDPP_DIR, PROSODY_DIR,
                    SOURCES)

SR = 16000
OUT_DIR = PROSODY_DIR


def extract(wav_path):
    f = {c: np.nan for c in FEATURE_COLS}
    try:
        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        if y.size < SR // 20:
            return f
        if float(np.max(np.abs(y))) <= 0.0:
            return f
        try:
            snd = parselmouth.Sound(str(wav_path))
        except Exception:
            snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)

        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        pv = pitch.selected_array["frequency"]
        pv = pv[pv > 0]
        if pv.size > 1:
            f["pitch_mean_hz"] = float(np.mean(pv))
            f["pitch_max_hz"] = float(np.max(pv))
            f["pitch_min_hz"] = float(np.min(pv))
            f["pitch_std_hz"] = float(np.std(pv))
            f["pitch_span_st"] = float(12 * np.log2(np.max(pv) / np.min(pv)))
            t = np.linspace(0, 1, pv.size)
            f["pitch_slope"] = float(np.polyfit(t, pv, 1)[0])

        try:
            _int = float(
                call(call(snd, "To Intensity", 100, 0), "Get mean", 0, 0, "dB"))
            f["intensity_mean_db"] = _int if np.isfinite(_int) and _int > -100 else np.nan
        except Exception:
            pass
        try:
            harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            h = call(harm, "Get mean", 0, 0)
            f["hnr_db"] = float(h) if np.isfinite(h) else np.nan
        except Exception:
            pass

        try:
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
            if call(pp, "Get number of points") > 2:
                f["jitter_pct"] = float(
                    call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3) * 100)
                f["shimmer_pct"] = float(
                    call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6) * 100)
        except Exception:
            pass

        f["duration_ms"] = float(len(y) / SR * 1000)
        f["spectral_centroid_hz"] = float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)[0]))
        rms = librosa.feature.rms(y=y)[0]
        if rms.size > 1:
            f["energy_slope"] = float(
                np.polyfit(np.linspace(0, 1, rms.size), rms, 1)[0])
        _thr = 0.01 * float(np.max(np.abs(y)))
        f["pause_ratio"] = float(np.mean(np.abs(y) < _thr))
    except Exception as e:
        print(f"    ! {Path(wav_path).name}: {e}")
    return f


def mp4_to_wav(src):
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(name)
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1",
                    str(tmp)], check=True, capture_output=True)
    return tmp


def items_nl():
    meta = pd.read_csv(DUTCH_METADATA)
    for _, r in meta.iterrows():
        yield (DUTCH_AUDIO / r["utterance_file"], str(r["label"]).lower(),
               str(r["participant_id"]), r["utterance_file"])


def items_en():
    meta = pd.read_csv(MUSTARDPP_DIR / "mustard++_text.csv")
    udir = MUSTARDPP_DIR / "final_utterance_videos"
    meta = meta[meta["Sarcasm"].notna()]
    for _, r in meta.iterrows():
        lab = "sarcastic" if float(r["Sarcasm"]) == 1.0 else "sincere"
        yield udir / f"{r['KEY']}.mp4", lab, str(r.get("SPEAKER", "unknown")), str(r["KEY"])


def items_de():
    data = json.loads((MUSAG_DIR / "dataset.json").read_text(encoding="utf-8"))
    for it in data:
        lab = "sarcastic" if it["label"] == "sarc" else "sincere"
        yield MUSAG_DIR / "segments" / f"{it['id']}.wav", lab, str(it.get("url", it["id"])), str(it["id"])


def items_zh():
    base = MCSD_DIR
    meta = pd.read_excel(base / "Files" / "MCSD_v1.xlsx")
    for _, r in meta.iterrows():
        lab = "sarcastic" if str(r["label"]).lower() in ("sarcastic", "1") else "sincere"
        yield base / "clips" / f"{r['utterance_id']}.wav", lab, str(r["speaker_id"]), str(r["utterance_id"])


CORPORA = {"nl": ("Dutch", items_nl), "en": ("MUStARD++", items_en),
           "de": ("MuSaG", items_de), "zh": ("MCSD", items_zh)}


def run(code, limit=None, resume=True):
    name, gen = CORPORA[code]
    cols = FEATURE_COLS + ["label", "speaker_id", "utt_id"]
    print(f"{name} ({code})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{code}_prosody.csv"
    part = OUT_DIR / f"{code}_prosody.partial.csv"

    done = set()
    if resume and part.exists():
        try:
            done = set(pd.read_csv(part)["utt_id"].astype(str))
            print(f"  resuming after {len(done)} rows")
        except Exception:
            done = set()
    elif part.exists():
        part.unlink()

    items = list(gen())
    if limit:
        items = items[:limit]

    missing, failed, written = [], 0, 0
    write_header = not part.exists()
    with io.open(part, "a", encoding="utf-8", newline="") as fh:
        for path, label, spk, uid in tqdm(items, desc=name):
            if str(uid) in done:
                continue
            if not path.exists():
                missing.append(str(uid))
                continue
            tmp = None
            try:
                if path.suffix.lower() in (".mp4", ".mkv", ".avi"):
                    tmp = mp4_to_wav(path)
                    feats = extract(tmp)
                else:
                    feats = extract(path)
            finally:
                if tmp is not None and tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
            if all(np.isnan(feats[c]) for c in FEATURE_COLS):
                failed += 1
            feats.update(label=label, speaker_id=spk, utt_id=uid)
            pd.DataFrame([feats])[cols].to_csv(fh, header=write_header, index=False)
            fh.flush()
            write_header = False
            written += 1

    df = pd.read_csv(part)[cols]
    df.to_csv(out, index=False)

    if missing:
        print(f"  {len(missing)} item(s) had no audio and were skipped, e.g. {missing[:5]}")
    if failed:
        print(f"  {failed} file(s) gave no features")
    print(f"  rows={len(df)}  speakers={df['speaker_id'].nunique()}  "
          f"sarcastic={(df['label'] == 'sarcastic').mean():.3f}  written {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N items (smoke test)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any .partial.csv and re-extract from scratch")
    a = ap.parse_args()
    for c in (list(CORPORA) if a.corpus == "all" else [a.corpus]):
        if not SOURCES[c].exists():
            print("%s: %s not found, skipped" % (c, SOURCES[c]))
            continue
        run(c, a.limit, resume=not a.no_resume)
