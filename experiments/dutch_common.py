import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import f1_score
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from config import (DUTCH_FEATURE_DIR, DUTCH_METADATA, FEATURE_DIR, PAPER_7, PROSODY_DIR,
                    TABLE_DIR)

OUT = TABLE_DIR
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, TAU, BATCH = 42, 0.07, 256
BS = chr(92)
BUDGETS = [5, 10, 20, 30, 50, 100]
META = {"label", "participant_id", "filename", "id", "name", "file"}

TIERS = [("7 Core Prosodic", "paper7", None),
         ("40 Prosodic+MFCC", "feat40", None),
         ("128 MFCC", "mfcc128", None),
         ("eGeMAPS", "egemaps", None),
         ("ComParE 2016", "compare", None),
         ("XLSR-53", "xlsr", None),
         ("emotion2vec\\_base", "emotion2vec", None),
         ("WavLM-Large MSP", "wavlm", None),
         ("Wav2Vec2-Large MSP", "wav2vec2", None)]
GEOMETRY = ["7 Core Prosodic", "emotion2vec\\_base", "Wav2Vec2-Large MSP"]


class Enc(nn.Module):
    def __init__(self, d, proj=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, proj))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=-1)


def supcon(z, y, tau=TAU):
    y = y.view(-1, 1)
    eye = torch.eye(len(y), device=z.device)
    mask = (y == y.T).float() - eye
    sim = (z @ z.T) / tau
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    logp = sim - torch.log((torch.exp(sim) * (1 - eye)).sum(1, keepdim=True) + 1e-9)
    return (-(mask * logp).sum(1) / mask.sum(1).clamp(min=1)).mean()


def ntxent(z1, z2, tau=TAU):
    n = z1.shape[0]
    z = nn.functional.normalize(torch.cat([z1, z2], 0), dim=-1)
    sim = (z @ z.T) / tau
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    sim.masked_fill_(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    return nn.CrossEntropyLoss()(sim, tgt)


def augment(X, sigma=0.1, drop=0.3):
    a = X + torch.randn_like(X) * sigma
    b = X + torch.randn_like(X) * sigma
    return (a * (torch.rand_like(X) > drop).float(),
            b * (torch.rand_like(X) > drop).float())


def train(Xtr, ytr, epochs, seed=SEED, labelled=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Enc(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.FloatTensor(Xtr).to(DEV)
    y = torch.LongTensor(ytr).to(DEV)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=DEV)
        for k in range(0, len(y), BATCH):
            b = perm[k:k + BATCH]
            if len(b) < 8 or len(torch.unique(y[b])) < 2:
                continue
            opt.zero_grad()
            if labelled:
                loss = supcon(m(X[b]), y[b])
            else:
                loss = ntxent(*[m(v) for v in augment(X[b])])
            loss.backward()
            opt.step()
    m.eval()
    return m


def plain(name):
    return name.replace(BS + "_", "_").replace("$" + BS + "to$", "-")


def project(m, *arrays):
    with torch.no_grad():
        return [m(torch.FloatTensor(a).to(DEV)).cpu().numpy() for a in arrays]


def svm_f1(Ztr, ytr, Zte, yte):
    pred = SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(Ztr, ytr).predict(Zte)
    return pred, float(f1_score(yte, pred, average="macro"))


def load(key):
    if key == "paper7":
        df = pd.read_csv(PROSODY_DIR / "nl_prosody.csv").dropna(subset=PAPER_7)
        y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
        return df[PAPER_7].values.astype(float), y, df["speaker_id"].astype(str).values
    csvs = {"feat40": "dutch_40_features.csv",
            "mfcc128": "dutch_128_mfcc_features.csv",
            "egemaps": "dutch_egemaps_features.csv",
            "compare": "dutch_compare_features.csv"}
    if key in csvs:
        df = pd.read_csv(DUTCH_FEATURE_DIR / csvs[key]).dropna()
        if "participant_id" not in df.columns:
            meta = pd.read_csv(DUTCH_METADATA)
            fn2p = dict(zip(meta["utterance_file"], meta["participant_id"]))
            df["participant_id"] = df["filename"].map(fn2p)
            df = df.dropna(subset=["participant_id"])
        cols = [c for c in df.columns
                if c not in META and pd.api.types.is_numeric_dtype(df[c])]
        y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
        return (df[cols].values.astype(float), y,
                df["participant_id"].astype(str).values)
    npz = {"xlsr": FEATURE_DIR / "xlsr/nl_xlsr_embeddings.npz",
           "emotion2vec": FEATURE_DIR / "emotion2vec/dutch_emotion2vec_embeddings.npz",
           "wavlm": FEATURE_DIR / "wavlm_msp/dutch_wavlm_msp_embeddings.npz",
           "wav2vec2": FEATURE_DIR / "wav2vec2_msp/dutch_wav2vec2_msp_embeddings.npz"}
    p = npz.get(key)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    meta = pd.read_csv(DUTCH_METADATA)
    if "labels" in d.files:
        raw = d["labels"]
        y = (np.array([1 if str(v).strip() == "sarcastic" else 0 for v in raw])
             if raw.dtype.kind in ("U", "S", "O") else raw.astype(int))
    else:
        fn2l = dict(zip(meta["utterance_file"], meta["label"]))
        y = np.array([1 if fn2l.get(f) == "sarcastic" else 0 for f in d["filenames"]])
    if "speakers" in d.files:
        spk = np.asarray(d["speakers"]).astype(str)
    else:
        fn2p = dict(zip(meta["utterance_file"], meta["participant_id"]))
        spk = np.array([str(fn2p.get(f, "unknown")) for f in d["filenames"]])
    return d["embeddings"].astype(np.float64), y, spk


def geometry(Zte, yte):
    intra = (np.mean(pdist(Zte[yte == 0])) + np.mean(pdist(Zte[yte == 1]))) / 2
    inter = np.mean(cdist(Zte[yte == 0], Zte[yte == 1]))
    return round(float(inter / intra) if intra > 0 else 0.0, 3)
