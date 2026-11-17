import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R
from budget_curves import fit_both

BUDGETS = [100, 150, 200]
DRAWS = 20
TIER = ("XLSR-53", "xlsr", "xlsr")


def main():
    shorter = pd.read_csv(R.OUT / "budget_curves.csv")
    shorter = shorter[shorter.tier == "XLSR-53"]
    rows, t0 = [], time.time()
    for code, cname in R.CORPORA.items():
        got = R.load(*TIER, code)
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
            if (~tr).sum() < 15 or len(np.unique(y[~tr])) < 2:
                continue
            for e in BUDGETS:
                a, b = fit_both(X[tr], y[tr], X[~tr], y[~tr], seed, e)
                te_s[e].append(a)
                tr_s[e].append(b)
        te = {e: float(np.mean(v)) for e, v in te_s.items()}
        trn = {e: float(np.mean(v)) for e, v in tr_s.items()}
        full = dict(te)
        prior = shorter[shorter.corpus == cname].set_index("epochs")
        for e in [5, 10, 20, 30, 50]:
            full[e] = prior.loc[e, "test_f1"]
        peak = max(full, key=full.get)
        print("%-9s 150: %.3f/%.3f  200: %.3f/%.3f  best %d epochs"
              % (cname, te[150], trn[150], te[200], trn[200], peak))
        for e in BUDGETS:
            rows.append(dict(corpus=cname, tier="XLSR-53", epochs=e,
                             test_f1=round(te[e], 4), train_f1=round(trn[e], 4),
                             test_sd=round(float(np.std(te_s[e], ddof=1)), 4),
                             n_draws=len(te_s[e]), overall_peak_epochs=peak))
    pd.DataFrame(rows).to_csv(R.OUT / "xlsr_ceiling.csv", index=False)
    print("%.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
