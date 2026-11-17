import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R
import contrastive_common as S

BUDGETS = [5, 10, 20, 30, 50, 100]
DRAWS = 20
MIN_TEST = 15


def ok(y, m):
    return m.sum() >= MIN_TEST and len(np.unique(y[m])) > 1


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for code, cname in S.CORPORA.items():
        for tier, sub, frag in S.TIERS:
            got = S.load(tier, sub, frag, code)
            if got is None:
                continue
            X, y, spk = got
            X = X.astype(np.float64)
            u0 = np.unique(spk)
            raws, sups, chosen = [], [], []
            for rep in range(DRAWS):
                seed = 1000 + rep
                u = u0.copy()
                np.random.RandomState(seed).shuffle(u)
                n_tr = max(1, int(len(u) * 0.8))
                tr = np.isin(spk, u[:n_tr])
                if not ok(y, ~tr):
                    continue
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[~tr])

                n_in = max(1, int(n_tr * 0.8))
                itr = np.isin(spk, u[:n_in])
                ival = np.isin(spk, u[n_in:n_tr])
                if ok(y, ival) and len(np.unique(y[itr])) > 1:
                    si = StandardScaler().fit(X[itr])
                    A, B = si.transform(X[itr]), si.transform(X[ival])
                    val = {}
                    for e in BUDGETS:
                        v, _ = S.supcon_svm(A, y[itr], B, y[ival], e, seed)
                        if v == v:
                            val[e] = v
                    best = max(val, key=val.get) if val else 30
                else:
                    best = 30

                raws.append(S.raw_svm(Xtr, y[tr], Xte, y[~tr]))
                v, _ = S.supcon_svm(Xtr, y[tr], Xte, y[~tr], best, seed)
                sups.append(v)
                chosen.append(best)

            if len(raws) < 3:
                continue
            r, s = np.array(raws), np.array(sups)
            d = (s - r) * 100
            try:
                p = float(wilcoxon(d).pvalue)
            except ValueError:
                p = float("nan")
            spread = ", ".join("%d:%d" % (e, n) for e, n in sorted(Counter(chosen).items()))
            print("%-9s %-20s raw %.4f  supcon %.4f  delta %+.2f pp  p %.4f"
                  % (cname, tier, r.mean(), s.mean(), d.mean(), p))
            rows.append(dict(corpus=cname, tier=tier,
                             raw_mean=round(float(r.mean()), 4),
                             supcon_mean=round(float(s.mean()), 4),
                             delta_pp=round(float(d.mean()), 2),
                             delta_sd_pp=round(float(d.std(ddof=1)), 2),
                             wilcoxon_p=round(p, 4) if p == p else None,
                             budgets_chosen=spread, n_draws=len(r),
                             deltas=";".join("%.2f" % v for v in d)))
    out = R.OUT / "contrastive_ablation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("%.1f min, written %s" % ((time.time() - t0) / 60, out))


if __name__ == "__main__":
    main()
