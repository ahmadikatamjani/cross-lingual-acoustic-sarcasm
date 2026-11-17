import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPORA, DEV, load

TIERS = [("7 Core Prosodic", None, None),
         ("emotion2vec", "emotion2vec", "emotion2vec"),
         ("Wav2Vec2-Large MSP", "wav2vec2_msp", "wav2vec2_msp"),
         ("WavLM-Large MSP", "wavlm_msp", "wavlm_msp")]


class Enc(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
                                 nn.Linear(h, 64))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def supcon_loss(z, y, tau=0.07):
    y = y.view(-1, 1)
    eye = torch.eye(len(y), device=z.device)
    mask = (y == y.T).float() - eye
    sim = z @ z.T / tau
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    logp = sim - torch.log((torch.exp(sim) * (1 - eye)).sum(1, keepdim=True) + 1e-9)
    return (-(mask * logp).sum(1) / mask.sum(1).clamp(min=1)).mean()


def raw_svm(Xtr, ytr, Xte, yte):
    pred = SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(Xtr, ytr).predict(Xte)
    return float(f1_score(yte, pred, average="macro"))


def supcon_svm(Xtr, ytr, Xte, yte, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Enc(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.FloatTensor(Xtr).to(DEV), torch.LongTensor(ytr).to(DEV)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=DEV)
        for k in range(0, len(yt), 256):
            b = perm[k:k + 256]
            if len(b) < 8 or len(torch.unique(yt[b])) < 2:
                continue
            opt.zero_grad()
            supcon_loss(m(Xt[b]), yt[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        etr = m(Xt).cpu().numpy()
        ete = m(torch.FloatTensor(Xte).to(DEV)).cpu().numpy()
    clf = SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(etr, ytr)
    return (float(f1_score(yte, clf.predict(ete), average="macro")),
            float(f1_score(ytr, clf.predict(etr), average="macro")))
