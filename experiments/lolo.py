import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R

OUT = R.OUT
CODES = ["nl", "en", "zh", "de"]
TIER = ("WavLM-Large MSP", "wavlm_msp", "wavlm_msp")
EPOCHS, SEED = 100, 42
SEEDS = [42, 43, 44, 45, 46]


def train(Xtr, ytr, seed=SEED, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    m = R.Enc(Z.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.FloatTensor(Z).to(R.DEV)
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
        etr = m(Xt).cpu().numpy()
    return sc, m, etr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = {}
    for c in CODES:
        got = R.load(*TIER, c)
        if got is not None:
            data[c] = got
    rows = []
    for c in CODES:
        if c not in data:
            continue
        Xtr = np.vstack([data[o][0] for o in data if o != c])
        ytr = np.concatenate([data[o][1] for o in data if o != c])
        Xte, yte, _ = data[c]
        f1s, accs, f1t, sincere = [], [], [], []
        for sd in SEEDS:
            sc, m, etr = train(Xtr, ytr, seed=sd)
            svm = SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(etr, ytr)
            with torch.no_grad():
                ete = m(torch.FloatTensor(sc.transform(Xte)).to(R.DEV)).cpu().numpy()
                tsc = StandardScaler().fit(Xte)
                etg = m(torch.FloatTensor(tsc.transform(Xte)).to(R.DEV)).cpu().numpy()
            pred = svm.predict(ete)
            f1s.append(float(f1_score(yte, pred, average="macro")))
            accs.append(float(accuracy_score(yte, pred)))
            sincere.append(float((pred == 0).mean()))
            f1t.append(float(f1_score(yte, svm.predict(etg), average="macro")))
        f1s, accs = np.array(f1s), np.array(accs)
        f1t, sincere = np.array(f1t), np.array(sincere)
        name = R.CORPORA[c]
        print("%-9s F1 %.4f +/- %.4f  standardised on the target %.4f"
              % (name, f1s.mean(), f1s.std(ddof=1), f1t.mean()))
        rows.append(dict(left_out=name, n_train=int(len(ytr)), n_test=int(len(yte)),
                         lolo_f1=round(float(f1s.mean()), 6),
                         lolo_f1_sd=round(float(f1s.std(ddof=1)), 6),
                         lolo_f1_min=round(float(f1s.min()), 6),
                         lolo_f1_max=round(float(f1s.max()), 6),
                         lolo_acc=round(float(accs.mean()), 6),
                         target_aligned_f1=round(float(f1t.mean()), 6),
                         target_aligned_sd=round(float(f1t.std(ddof=1)), 6),
                         target_aligned_delta_pp=round(float((f1t.mean() - f1s.mean()) * 100), 3),
                         pred_sincere_rate=round(float(sincere.mean()), 4),
                         n_seeds=len(SEEDS), epochs=EPOCHS,
                         scores=";".join("%.6f" % v for v in f1s)))
    pd.DataFrame(rows).to_csv(OUT / "lolo.csv", index=False)
    print("%.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
