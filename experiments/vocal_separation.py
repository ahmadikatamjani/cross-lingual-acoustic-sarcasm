import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[0] / "extraction"))
sys.path.insert(0, str(HERE))
from config import ALL_PROSODIC, FEATURE_DIR, PAPER_7, PROSODY_DIR, TABLE_DIR
import common as R
import monolingual as N
import prosody as E

SEP = FEATURE_DIR / "vocal_separated"
WAVDIR = SEP / "en_wav"


def items():
    return [(p, lab, spk, key) for p, lab, spk, key in E.items_en() if Path(p).exists()]


def separate_signal(y, sr):
    S_full, phase = librosa.magphase(librosa.stft(y))
    t = S_full.shape[1]
    width = min(int(librosa.time_to_frames(2, sr=sr)), (t - 1) // 2 - 1)
    if width < 1:
        return y
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine", width=width)
    S_filter = np.minimum(S_full, S_filter)
    mask_v = librosa.util.softmask(S_full - S_filter, 10 * S_filter, power=2)
    return librosa.istft(mask_v * S_full * phase, length=len(y))


def separate():
    WAVDIR.mkdir(parents=True, exist_ok=True)
    its = items()
    t0 = time.time()
    for k, (p, lab, spk, key) in enumerate(its):
        tmp = E.mp4_to_wav(Path(p))
        try:
            y, sr = librosa.load(str(tmp), sr=None, mono=True)
        finally:
            tmp.unlink(missing_ok=True)
        sf.write(str(WAVDIR / ("%s.wav" % key)), separate_signal(y, sr), sr, subtype="FLOAT")
        if k % 100 == 0:
            print("separated %4d/%d  (%.1f min)" % (k + 1, len(its), (time.time() - t0) / 60), flush=True)


def wavlm_model():
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    path = hf_hub_download("tiantiaf/wavlm-large-msp-podcast-emotion-dim", "model.safetensors")
    state = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("backbone_model."):
                state[k[len("backbone_model."):]] = f.get_tensor(k)
    base = snapshot_download("microsoft/wavlm-large")
    m = WavLMModel.from_pretrained(base)
    m.load_state_dict(state, strict=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return Wav2Vec2FeatureExtractor.from_pretrained(base), m.eval().to(dev), dev


def embed(proc, model, dev, audio):
    inp = proc(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()


def extract():
    stored = np.load(FEATURE_DIR / "wavlm_msp" / "mustardpp_wavlm_msp_embeddings.npz", allow_pickle=True)
    proc, model, dev = wavlm_model()
    sep = [embed(proc, model, dev, librosa.load(str(WAVDIR / ("%s.wav" % k)), sr=16000)[0])
           for k in (str(k) for k in stored["keys"])]
    np.savez(SEP / "en_wavlm_msp_separated.npz", embeddings=np.array(sep), labels=stored["labels"],
             keys=stored["keys"], speakers=stored["speakers"])

    pub = pd.read_csv(PROSODY_DIR / "en_prosody.csv")
    ps = pd.DataFrame([E.extract(WAVDIR / ("%s.wav" % u)) for u in pub.utt_id])[ALL_PROSODIC]
    for c in ("label", "speaker_id", "utt_id"):
        ps[c] = pub[c].values
    ps.to_csv(SEP / "en_prosody_separated.csv", index=False)


def load_separated(tier):
    if tier == "WavLM-Large MSP":
        d = np.load(SEP / "en_wavlm_msp_separated.npz", allow_pickle=True)
        return (d["embeddings"].astype(np.float64), d["labels"].astype(int),
                np.asarray(d["speakers"]).astype(str), None)
    df = pd.read_csv(SEP / "en_prosody_separated.csv")
    pub = pd.read_csv(PROSODY_DIR / "en_prosody.csv")
    keep = df[PAPER_7].notna().all(1) & pub[PAPER_7].notna().all(1)
    df = df[keep]
    y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
    return df[PAPER_7].values.astype(float), y, df["speaker_id"].astype(str).values, keep.values


def bench():
    from scipy.stats import wilcoxon
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tier, sub, frag in [("WavLM-Large MSP", "wavlm_msp", "wavlm_msp"), ("7 Core Prosodic", None, None)]:
        Xo, yo, so = R.load(tier, sub, frag, "en")
        Xs, ys, ss, keep = load_separated(tier)
        if keep is not None:
            pub = pd.read_csv(PROSODY_DIR / "en_prosody.csv")
            ok = keep[pub[PAPER_7].notna().all(1).values]
            Xo, yo, so = Xo[ok], yo[ok], so[ok]
        if not (np.array_equal(yo, ys) and np.array_equal(so, ss)):
            raise SystemExit("%s: labels or speakers differ between the two versions" % tier)
        so_scores, so_ch, so_dr = N.nested_cell(Xo.astype(np.float64), yo, so)
        ss_scores, ss_ch, ss_dr = N.nested_cell(Xs.astype(np.float64), ys, ss)
        if so_dr != ss_dr:
            raise SystemExit("%s: the two versions kept different draws" % tier)
        for k, r in enumerate(so_dr):
            rows.append(dict(tier=tier, draw=r, original=so_scores[k], separated=ss_scores[k],
                             budget_original=so_ch[k], budget_separated=ss_ch[k]))
        d = (ss_scores - so_scores) * 100
        print("%-16s original %.3f  separated %.3f  %+.2f pp, p = %.3g"
              % (tier, so_scores.mean(), ss_scores.mean(), d.mean(),
                 wilcoxon(ss_scores, so_scores).pvalue))
    pd.DataFrame(rows).to_csv(TABLE_DIR / "vocal_separation.csv", index=False)


if __name__ == "__main__":
    SEP.mkdir(parents=True, exist_ok=True)
    {"separate": separate, "extract": extract, "bench": bench}[sys.argv[1]]()
