import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DUTCH_AUDIO, DUTCH_FEATURE_DIR, DUTCH_METADATA, FEATURE_DIR, MCSD_DIR,
                    MUSAG_DIR, MUSTARDPP_DIR)

SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEM = {"nl": "dutch", "en": "mustardpp", "zh": "mcsd", "de": "musag"}
WAV2VEC2_MSP = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
WAVLM_MSP = "tiantiaf/wavlm-large-msp-podcast-emotion-dim"
XLSR_ID, XLSR_REVISION = "facebook/wav2vec2-large-xlsr-53", "c3f9d884181a224a6ac87bf8885c84d1cff3384f"


def targets(code):
    if code == "nl":
        for _, r in pd.read_csv(DUTCH_METADATA).iterrows():
            p = DUTCH_AUDIO / r["utterance_file"]
            if p.exists():
                yield p, r["label"], r["utterance_file"], r["participant_id"]
    elif code == "en":
        meta = pd.read_csv(MUSTARDPP_DIR / "mustard++_text.csv")
        for _, r in meta[meta["Sarcasm"].notna()].iterrows():
            p = MUSTARDPP_DIR / "final_utterance_videos" / ("%s.mp4" % r["KEY"])
            if p.exists():
                yield p, int(r["Sarcasm"]), r["KEY"], r["SPEAKER"]
    elif code == "zh":
        meta = pd.read_excel(MCSD_DIR / "Files" / "MCSD_v1.xlsx")
        labels = {"sarcastic": 1, "not_sarcastic": 0}
        for _, r in meta.iterrows():
            p = MCSD_DIR / "clips" / ("%s.wav" % r["utterance_id"])
            if p.exists() and p.stat().st_size > 1000 and r["label"] in labels:
                yield p, labels[r["label"]], r["utterance_id"], r["speaker_id"]
    else:
        meta = pd.read_csv(MUSAG_DIR / "dataset.csv")
        labels = {"sarc": 1, "non-sarc": 0}
        for _, r in meta.iterrows():
            p = MUSAG_DIR / "segments" / ("%s.wav" % r["id"])
            if p.exists() and r["label"] in labels:
                yield p, labels[r["label"]], str(r["id"]), r["url"]


def save(path, code, embs, labels, keys, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(embeddings=np.array(embs, dtype=np.float32), labels=np.array(labels))
    if code != "nl":
        out["keys"] = np.array(keys)
    out["sources" if code == "de" else "speakers"] = np.array(groups)
    np.savez(path, **out)
    print("%s: %d utterances -> %s" % (code, len(embs), path))


def wavlm_msp():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    state = {}
    with safe_open(hf_hub_download(WAVLM_MSP, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("backbone_model."):
                state[k[len("backbone_model."):]] = f.get_tensor(k)
    model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    model.load_state_dict(state, strict=False)
    return Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large"), model


def wav2vec2_msp():
    from transformers import AutoFeatureExtractor, AutoModel
    return AutoFeatureExtractor.from_pretrained(WAV2VEC2_MSP), AutoModel.from_pretrained(WAV2VEC2_MSP)


def ssl(name, code):
    processor, model = {"wavlm_msp": wavlm_msp, "wav2vec2_msp": wav2vec2_msp}[name]()
    model = model.to(DEVICE).eval()
    embs, labels, keys, groups = [], [], [], []
    for p, lab, key, grp in tqdm(list(targets(code)), desc="%s %s" % (name, code)):
        try:
            audio, _ = librosa.load(str(p), sr=SR, mono=True)
            inputs = processor(audio, sampling_rate=SR, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            embs.append(out.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy())
        except Exception as e:
            print("%s: %s" % (p.name, e))
            continue
        labels.append(lab)
        keys.append(key)
        groups.append(grp)
    save(FEATURE_DIR / name / ("%s_%s_embeddings.npz" % (STEM[code], name)), code, embs, labels, keys, groups)


def emotion2vec(code):
    from funasr import AutoModel
    model = AutoModel(model="iic/emotion2vec_base", disable_update=True,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    embs, labels, keys, groups = [], [], [], []
    for p, lab, key, grp in tqdm(list(targets(code)), desc="emotion2vec %s" % code):
        try:
            res = model.generate(str(p), output_dir=None, granularity="utterance", extract_embedding=True)
            embs.append(np.array(res[0]["feats"]).flatten())
        except Exception as e:
            print("%s: %s" % (p.name, e))
            continue
        labels.append(lab)
        keys.append(key)
        groups.append(grp)
    save(FEATURE_DIR / "emotion2vec" / ("%s_emotion2vec_embeddings.npz" % STEM[code]),
         code, embs, labels, keys, groups)


def opensmile_set(name, code):
    import opensmile
    smile = opensmile.Smile(
        feature_set={"egemaps": opensmile.FeatureSet.eGeMAPSv02,
                     "compare": opensmile.FeatureSet.ComParE_2016}[name],
        feature_level=opensmile.FeatureLevel.Functionals)
    if code == "nl":
        rows = []
        for p, lab, key, grp in tqdm(list(targets(code)), desc="%s nl" % name):
            feats = smile.process_file(str(p)).iloc[0].to_dict()
            if name == "egemaps":
                feats.update(participant_id=grp, label=lab, utterance_file=key)
            else:
                feats.update(id=Path(key).stem, filename=key, label=lab, participant_id=grp)
            rows.append(feats)
        DUTCH_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        out = DUTCH_FEATURE_DIR / ("dutch_%s_features.csv" % name)
        pd.DataFrame(rows).to_csv(out, index=False)
        print("nl: %d utterances -> %s" % (len(rows), out))
        return
    embs, labels, keys, groups = [], [], [], []
    for p, lab, key, grp in tqdm(list(targets(code)), desc="%s %s" % (name, code)):
        try:
            audio, sr = librosa.load(str(p), sr=SR)
            embs.append(smile.process_signal(audio, sr).values[0])
        except Exception as e:
            print("%s: %s" % (p.name, e))
            continue
        labels.append(lab)
        keys.append(key)
        groups.append(grp)
    save(FEATURE_DIR / name / ("%s_%s_embeddings.npz" % (STEM[code], name)), code, embs, labels, keys, groups)


def xlsr_items(code):
    if code == "nl":
        for p, lab, key, grp in targets("nl"):
            yield p, lab, grp, key
    elif code == "en":
        meta = pd.read_csv(MUSTARDPP_DIR / "mustard++_text.csv")
        for _, r in meta[meta["Sarcasm"].notna()].iterrows():
            yield (MUSTARDPP_DIR / "final_utterance_videos" / ("%s.mp4" % r["KEY"]),
                   1 if float(r["Sarcasm"]) == 1.0 else 0, str(r.get("SPEAKER", "unknown")), None)
    elif code == "de":
        for it in json.loads((MUSAG_DIR / "dataset.json").read_text(encoding="utf-8")):
            yield (MUSAG_DIR / "segments" / ("%s.wav" % it["id"]),
                   1 if it["label"] == "sarc" else 0, str(it.get("url", it["id"])), None)
    else:
        for _, r in pd.read_excel(MCSD_DIR / "Files" / "MCSD_v1.xlsx").iterrows():
            yield (MCSD_DIR / "clips" / ("%s.wav" % r["utterance_id"]),
                   1 if str(r["label"]).strip().lower() == "sarcastic" else 0, str(r["speaker_id"]), None)


def to_wav(src):
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), name],
                   check=True, capture_output=True)
    return Path(name)


def xlsr(code):
    from huggingface_hub import snapshot_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    local = snapshot_download(XLSR_ID, revision=XLSR_REVISION)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(local)
    model = Wav2Vec2Model.from_pretrained(local).to(DEVICE).eval()
    E, Y, S, F = [], [], [], []
    with torch.no_grad():
        for path, y, spk, fname in tqdm(list(xlsr_items(code)), desc="xlsr %s" % code):
            if not path.exists():
                continue
            tmp = to_wav(path) if path.suffix.lower() == ".mp4" else None
            try:
                wav, _ = librosa.load(str(tmp or path), sr=SR, mono=True)
                if wav.size < SR // 20:
                    continue
                iv = fe(wav, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
                E.append(model(iv).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy())
            except Exception as e:
                print("%s: %s" % (path.name, e))
                continue
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            Y.append(y)
            S.append(spk)
            F.append(fname)
    out = FEATURE_DIR / "xlsr" / ("%s_xlsr_embeddings.npz" % code)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {"filenames": np.array(F)} if code == "nl" else {}
    np.savez_compressed(out, embeddings=np.stack(E).astype(np.float32), labels=np.array(Y),
                        speakers=np.array(S, dtype=object), **extra)
    print("%s: %d utterances -> %s" % (code, len(E), out))


def run(name, code):
    if name in ("wavlm_msp", "wav2vec2_msp"):
        ssl(name, code)
    elif name == "emotion2vec":
        emotion2vec(code)
    elif name in ("egemaps", "compare"):
        opensmile_set(name, code)
    else:
        xlsr(code)


if __name__ == "__main__":
    names = ["wav2vec2_msp", "wavlm_msp", "emotion2vec", "egemaps", "compare", "xlsr"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=names + ["all"], default="all")
    ap.add_argument("--corpus", choices=list(STEM) + ["all"], default="all")
    a = ap.parse_args()
    for n in (names if a.features == "all" else [a.features]):
        for c in (list(STEM) if a.corpus == "all" else [a.corpus]):
            run(n, c)
