import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R

EPOCHS, BATCH, LR, DRAWS = 100, 128, 1e-3, 20
CORPORA = [("nl", "Dutch"), ("en", "English (MUStARD++)"),
           ("zh", "Mandarin (MCSD)"), ("de", "German (MuSaG)")]
TIERS = [("7 Core Prosodic", None, None),
         ("ComParE 2016", "compare", "compare"),
         ("Wav2Vec2-Large MSP", "wav2vec2_msp", "wav2vec2_msp")]


def arm(X, y, tr, te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte = y[tr], y[te]
    m = R.Enc(Xtr.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    Xt = torch.FloatTensor(Xtr).to(R.DEV)
    yt = torch.LongTensor(ytr).to(R.DEV)
    m.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(yt), device=R.DEV)
        for k in range(0, len(yt), BATCH):
            b = perm[k:k + BATCH]
            if len(b) < 4 or len(torch.unique(yt[b])) < 2:
                continue
            opt.zero_grad()
            R.supcon(m(Xt[b]), yt[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        ztr = m(Xt).cpu().numpy()
        zte = m(torch.FloatTensor(Xte).to(R.DEV)).cpu().numpy()
    sil = silhouette_score(zte, yte) if len(np.unique(yte)) > 1 else float("nan")
    clf = SVC(kernel="rbf", C=1.0, class_weight="balanced").fit(ztr, ytr)
    return float(sil), float(f1_score(yte, clf.predict(zte), average="macro"))


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for code, cname in CORPORA:
        for tier, sub, frag in TIERS:
            got = R.load(tier, sub, frag, code)
            if got is None:
                print("%-20s %-19s missing" % (cname, tier))
                continue
            X, y, spk = got
            X = X.astype(np.float64)
            u0 = np.unique(spk)
            cs, cf, ls, lf = [], [], [], []
            for r in range(DRAWS):
                seed = 1000 + r
                rng = np.random.RandomState(seed)
                u = u0.copy()
                rng.shuffle(u)
                mask = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
                if (~mask).sum() < 15 or len(np.unique(y[~mask])) < 2:
                    continue
                s1, f1c = arm(X, y, np.where(mask)[0], np.where(~mask)[0], seed)
                itr, ite = train_test_split(np.arange(len(y)), test_size=0.2,
                                            random_state=seed, stratify=y)
                s2, f1l = arm(X, y, itr, ite, seed)
                cs.append(s1)
                cf.append(f1c)
                ls.append(s2)
                lf.append(f1l)

            diff = np.array(lf) - np.array(cf)
            p = wilcoxon(diff).pvalue if len(diff) > 5 and np.any(diff != 0) else float("nan")
            print("%-20s %-19s clean %.3f  leaked %.3f  delta %+.1f pp  p %.3f"
                  % (cname, tier, np.mean(cf), np.mean(lf), diff.mean() * 100, p))
            rows.append(dict(
                corpus=cname, feature_set=tier, n_draws=len(cf),
                clean_sil=round(float(np.nanmean(cs)), 3),
                leaked_sil=round(float(np.nanmean(ls)), 3),
                clean_f1=round(float(np.mean(cf)), 4), clean_sd=round(float(np.std(cf, ddof=1)), 4),
                leaked_f1=round(float(np.mean(lf)), 4), leaked_sd=round(float(np.std(lf, ddof=1)), 4),
                delta_pp=round(float(diff.mean()) * 100, 1),
                delta_sd_pp=round(float(diff.std(ddof=1)) * 100, 1),
                p_wilcoxon=round(float(p), 4) if p == p else None,
                clean_min=round(min(cf), 4), clean_max=round(max(cf), 4),
                n=len(y), dim=X.shape[1], n_speakers=len(u0)))
    out = R.OUT / "leakage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("%.1f min, written %s" % ((time.time() - t0) / 60, out))


if __name__ == "__main__":
    main()
