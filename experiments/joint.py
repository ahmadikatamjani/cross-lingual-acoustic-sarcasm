import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R

DRAWS = 20
CODES = ["nl", "en", "zh", "de"]
TIERS = [("WavLM-Large MSP", "wavlm_msp", "wavlm_msp"),
         ("Wav2Vec2-Large MSP", "wav2vec2_msp", "wav2vec2_msp"),
         ("ComParE 2016", "compare", "compare"),
         ("emotion2vec", "emotion2vec", "emotion2vec"),
         ("eGeMAPS", "egemaps", "egemaps"),
         ("7 Core Prosodic", None, None)]


def fit_eval(Xtr, ytr, Xte, yte, epochs, wd, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = R.Enc(Xtr.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    Xt = torch.FloatTensor(Xtr).to(R.DEV)
    yt = torch.LongTensor(ytr).to(R.DEV)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=R.DEV)
        for k in range(0, len(yt), 256):
            b = perm[k:k + 256]
            if len(b) < 8 or len(torch.unique(yt[b])) < 2:
                continue
            opt.zero_grad()
            R.supcon(m(Xt[b]), yt[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        ztr = m(Xt).cpu().numpy()
        zte = m(torch.FloatTensor(Xte).to(R.DEV)).cpu().numpy()
    clf = SVC(kernel="rbf", C=1.0, class_weight="balanced").fit(ztr, ytr)
    return float(f1_score(yte, clf.predict(zte), average="macro"))


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for tier, sub, frag in TIERS:
        parts = {c: R.load(tier, sub, frag, c) for c in CODES}
        if any(v is None for v in parts.values()):
            print("%-22s missing" % tier)
            continue
        n_pooled = sum(len(parts[c][1]) for c in CODES)
        sis = []
        for r in range(DRAWS):
            seed = 1000 + r
            trX, trY, teX, teY = [], [], [], []
            for c in CODES:
                Xc, yc, spk = parts[c]
                u = np.unique(spk)
                np.random.RandomState(seed).shuffle(u)
                mask = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
                trX.append(Xc[mask])
                trY.append(yc[mask])
                teX.append(Xc[~mask])
                teY.append(yc[~mask])
            sis.append(fit_eval(np.vstack(trX).astype(np.float64), np.concatenate(trY),
                                np.vstack(teX).astype(np.float64), np.concatenate(teY),
                                100, 1e-4, seed))
        print("%-22s %.3f +/- %.3f" % (tier, np.mean(sis), np.std(sis, ddof=1)))
        rows.append(dict(feature_set=tier,
                         si_f1=round(float(np.mean(sis)), 4),
                         si_sd=round(float(np.std(sis, ddof=1)), 4),
                         si_min=round(min(sis), 4), si_max=round(max(sis), 4),
                         n_pooled=n_pooled, dim=parts[CODES[0]][0].shape[1],
                         si_scores=";".join("%.4f" % v for v in sis)))
    df = pd.DataFrame(rows).sort_values("si_f1", ascending=False).reset_index(drop=True)
    out = R.OUT / "joint.csv"
    df.to_csv(out, index=False)
    print("%.1f min, written %s" % ((time.time() - t0) / 60, out))


if __name__ == "__main__":
    main()
