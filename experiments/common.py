import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CORPORA, DUTCH_FEATURE_DIR, DUTCH_METADATA, FEATURE_DIR, PAPER_7,
                    PROSODY_DIR, TABLE_DIR)

DATA = FEATURE_DIR
OUT = TABLE_DIR
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEM = {"nl": "dutch", "en": "mustardpp", "zh": "mcsd", "de": "musag"}
TIERS = [("7 Core Prosodic", None, None),
         ("eGeMAPS", "egemaps", "egemaps"),
         ("ComParE 2016", "compare", "compare"),
         ("XLSR-53", "xlsr", "xlsr"),
         ("emotion2vec", "emotion2vec", "emotion2vec"),
         ("Wav2Vec2-Large MSP", "wav2vec2_msp", "wav2vec2_msp"),
         ("WavLM-Large MSP", "wavlm_msp", "wavlm_msp")]
DUTCH_CSV = {"egemaps": "dutch_egemaps_features.csv", "compare": "dutch_compare_features.csv"}
META = {"label", "participant_id", "filename", "id", "name", "file"}


class Enc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, 64))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def supcon(z, y, tau=0.07):
    y = y.view(-1, 1)
    eye = torch.eye(len(y), device=z.device)
    mask = (y == y.T).float() - eye
    sim = z @ z.T / tau
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    logp = sim - torch.log((torch.exp(sim) * (1 - eye)).sum(1, keepdim=True) + 1e-9)
    return (-(mask * logp).sum(1) / mask.sum(1).clamp(min=1)).mean()


def fit_eval(Xtr, ytr, Xte, yte, seed, epochs=30):
    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
        return float("nan")
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = Enc(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.FloatTensor(Xtr).to(DEV)
    yt = torch.LongTensor(ytr).to(DEV)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=DEV)
        for k in range(0, len(yt), 256):
            b = perm[k:k + 256]
            if len(b) < 8 or len(torch.unique(yt[b])) < 2:
                continue
            opt.zero_grad()
            supcon(m(Xt[b]), yt[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        etr = m(Xt).cpu().numpy()
        ete = m(torch.FloatTensor(Xte).to(DEV)).cpu().numpy()
    pred = SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(etr, ytr).predict(ete)
    return float(f1_score(yte, pred, average="macro"))


def load(tier, subdir, frag, code):
    if tier == "7 Core Prosodic":
        p = PROSODY_DIR / ("%s_prosody.csv" % code)
        if not p.exists():
            return None
        df = pd.read_csv(p).dropna(subset=PAPER_7)
        y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
        return df[PAPER_7].values.astype(float), y, df["speaker_id"].astype(str).values
    if code == "nl" and frag in DUTCH_CSV:
        p = DUTCH_FEATURE_DIR / DUTCH_CSV[frag]
        if not p.exists():
            return None
        df = pd.read_csv(p).dropna()
        cols = [c for c in df.columns
                if c not in META and pd.api.types.is_numeric_dtype(df[c])]
        y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
        return df[cols].values.astype(float), y, df["participant_id"].astype(str).values
    if subdir == "xlsr":
        hits = [c for c in [DATA / "xlsr" / ("%s_xlsr_embeddings.npz" % code)] if c.exists()]
    else:
        hits = list((DATA / subdir).glob("%s*%s*.npz" % (STEM[code], frag)))
    if not hits:
        return None
    d = np.load(hits[0], allow_pickle=True)
    raw = d["labels"]
    y = (np.array([1 if str(v).strip() == "sarcastic" else 0 for v in raw])
         if raw.dtype.kind in ("U", "S", "O") else raw.astype(int))
    if "speakers" in d.files:
        spk = np.asarray(d["speakers"]).astype(str)
    elif "sources" in d.files:
        spk = np.asarray(d["sources"]).astype(str)
    elif "filenames" in d.files and code == "nl":
        meta = pd.read_csv(DUTCH_METADATA)
        fn2p = dict(zip(meta["utterance_file"], meta["participant_id"]))
        spk = np.array([str(fn2p.get(f, "unknown")) for f in d["filenames"]])
    else:
        return None
    return d["embeddings"].astype(np.float64), y, spk
