import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R

BUDGETS = [5, 10, 20, 30, 50, 100]
DRAWS = 20
MIN_TEST = 15


def usable(y, mask):
    return mask.sum() >= MIN_TEST and len(np.unique(y[mask])) > 1


def nested_cell(X, y, spk):
    u0 = np.unique(spk)
    scores, chosen = [], []
    for r in range(DRAWS):
        seed = 1000 + r
        u = u0.copy()
        np.random.RandomState(seed).shuffle(u)
        n_tr = max(1, int(len(u) * 0.8))
        tr_spk, te_spk = u[:n_tr], u[n_tr:]
        tr, te = np.isin(spk, tr_spk), np.isin(spk, te_spk)
        if not usable(y, te):
            continue

        n_in = max(1, int(n_tr * 0.8))
        itr = np.isin(spk, tr_spk[:n_in])
        ival = np.isin(spk, tr_spk[n_in:])
        if not usable(y, ival) or len(np.unique(y[itr])) < 2:
            best = BUDGETS[len(BUDGETS) // 2]
        else:
            val = {}
            for e in BUDGETS:
                v = R.fit_eval(X[itr], y[itr], X[ival], y[ival], seed, epochs=e)
                if v == v:
                    val[e] = v
            best = max(val, key=val.get) if val else BUDGETS[len(BUDGETS) // 2]

        v = R.fit_eval(X[tr], y[tr], X[te], y[te], seed, epochs=best)
        if v == v:
            scores.append(v)
            chosen.append(best)
    return np.array(scores), chosen


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for code in R.CORPORA:
        for tier, sub, frag in R.TIERS:
            got = R.load(tier, sub, frag, code)
            if got is None:
                continue
            X, y, spk = got
            s, chosen = nested_cell(X.astype(np.float64), y, spk)
            if not len(s):
                print("%-9s %-20s no viable draws" % (R.CORPORA[code], tier))
                continue
            mu, sd = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0
            spread = ", ".join("%d:%d" % (e, n) for e, n in sorted(Counter(chosen).items()))
            print("%-9s %-20s %.3f +/- %.3f  %s" % (R.CORPORA[code], tier, mu, sd, spread))
            rows.append(dict(corpus=R.CORPORA[code], tier=tier,
                             nested_mean=round(mu, 4), nested_sd=round(sd, 4),
                             nested_min=round(float(s.min()), 4),
                             nested_max=round(float(s.max()), 4),
                             budgets_chosen=spread, n_draws=len(s),
                             scores=";".join("%.4f" % v for v in s)))
    out = R.OUT / "monolingual.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("%.1f min, written %s" % ((time.time() - t0) / 60, out))


if __name__ == "__main__":
    main()
