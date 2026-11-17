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

BUDGETS = [5, 10, 20, 30, 50, 100]
DRAWS = 20
MIN_TEST = 15


def fit_both(Xtr, ytr, Xte, yte, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = R.Enc(Xtr.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
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
    return (float(f1_score(yte, clf.predict(zte), average="macro")),
            float(f1_score(ytr, clf.predict(ztr), average="macro")))


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for code in R.CORPORA:
        cname = R.CORPORA[code]
        for tier, sub, frag in R.TIERS:
            got = R.load(tier, sub, frag, code)
            if got is None:
                continue
            X, y, spk = got
            X = X.astype(np.float64)
            u0 = np.unique(spk)
            te_s = {e: [] for e in BUDGETS}
            tr_s = {e: [] for e in BUDGETS}
            for r in range(DRAWS):
                seed = 1000 + r
                u = u0.copy()
                np.random.RandomState(seed).shuffle(u)
                tr = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
                if (~tr).sum() < MIN_TEST or len(np.unique(y[~tr])) < 2:
                    continue
                for e in BUDGETS:
                    a, b = fit_both(X[tr], y[tr], X[~tr], y[~tr], seed, e)
                    te_s[e].append(a)
                    tr_s[e].append(b)
            if not te_s[BUDGETS[0]]:
                continue
            te = {e: float(np.mean(v)) for e, v in te_s.items()}
            trn = {e: float(np.mean(v)) for e, v in tr_s.items()}
            peak = max(BUDGETS, key=lambda e: te[e])
            print("%-9s %-20s %s" % (cname, tier, "  ".join("%d: %.3f/%.3f" % (e, te[e], trn[e])
                                                          for e in BUDGETS)))
            for e in BUDGETS:
                rows.append(dict(corpus=cname, tier=tier, epochs=e,
                                 test_f1=round(te[e], 4), train_f1=round(trn[e], 4),
                                 test_sd=round(float(np.std(te_s[e], ddof=1)), 4),
                                 is_peak=(e == peak), n_draws=len(te_s[e])))
    df = pd.DataFrame(rows)
    df.to_csv(R.OUT / "budget_curves.csv", index=False)
    print("%.1f min" % ((time.time() - t0) / 60))
    nested = R.OUT / "monolingual.csv"
    if nested.exists():
        peak = df[df.is_peak].set_index(["corpus", "tier"]).test_f1
        m = pd.read_csv(nested).set_index(["corpus", "tier"]).nested_mean
        d = ((m - peak.reindex(m.index)) * 100).rename("nested_minus_test_selected_pp")
        d.reset_index().to_csv(R.OUT / "budget_selection.csv", index=False)
        print("budget chosen on the test speakers: nested selection lower by %.2f pp on average, "
              "lower in %d of %d cells, by %.2f pp at most" % (-d.mean(), (d < 0).sum(), d.notna().sum(), -d.min()))


if __name__ == "__main__":
    main()
