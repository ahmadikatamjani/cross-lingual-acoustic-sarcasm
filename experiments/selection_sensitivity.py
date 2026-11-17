import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import ALL_PROSODIC, PAPER_7, PROSODY_DIR, TABLE_DIR
import common as R

SRC = PROSODY_DIR
OUT = TABLE_DIR / "selection_sensitivity.csv"
BUDGETS = [5, 10, 20, 30, 50, 100]
DRAWS, MIN_TEST = 20, 15


def usable(y, mask):
    return mask.sum() >= MIN_TEST and len(np.unique(y[mask])) > 1


def abs_d(X, y, rows):
    a, b = X[rows & (y == 1)], X[rows & (y == 0)]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1))
                 / (len(a) + len(b) - 2))
    return np.abs((a.mean(0) - b.mean(0)) / np.where(sp == 0, np.nan, sp))


def train_only_seven(X, y, rows):
    d = np.nan_to_num(abs_d(X, y, rows))
    return list(np.argsort(-d, kind="stable")[:7])


def holm(p):
    o = np.argsort(p)
    m, out, prev = len(p), np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        prev = max(prev, min(1.0, (m - r) * p[i]))
        out[i] = prev
    return out


def nested(X, y, spk, cols_for):
    u0 = np.unique(spk)
    scores, sets, budgets = [], [], []
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
            c = cols_for(itr)
            val = {}
            for e in BUDGETS:
                v = R.fit_eval(X[itr][:, c], y[itr], X[ival][:, c], y[ival], seed, epochs=e)
                if v == v:
                    val[e] = v
            best = max(val, key=val.get) if val else BUDGETS[len(BUDGETS) // 2]
        c = cols_for(tr)
        v = R.fit_eval(X[tr][:, c], y[tr], X[te][:, c], y[te], seed, epochs=best)
        if v == v:
            scores.append(v)
            sets.append("|".join(ALL_PROSODIC[i] for i in c))
            budgets.append(best)
    return np.array(scores), sets, budgets


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixed_cols = [ALL_PROSODIC.index(f) for f in PAPER_7]
    rows, t0 = [], time.time()
    for code, name in R.CORPORA.items():
        df = pd.read_csv(SRC / ("%s_prosody.csv" % code)).dropna(subset=PAPER_7)
        if df[ALL_PROSODIC].isna().any(axis=1).any():
            raise SystemExit("%s: missing values outside the seven" % name)
        y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
        spk = df["speaker_id"].astype(str).values
        X = df[ALL_PROSODIC].values.astype(np.float64)

        fixed, _, fixed_bud = nested(X, y, spk, lambda rows: fixed_cols)
        train, sets, bud = nested(X, y, spk, lambda rows: train_only_seven(X, y, rows))
        d = train - fixed
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        same = sum(set(s.split("|")) == set(PAPER_7) for s in sets)
        common = Counter(sets).most_common(1)[0]
        print("%-9s fixed %.4f  train-only %.4f  delta %+.2f pp  p = %.3f"
              % (name, fixed.mean(), train.mean(), d.mean() * 100, p), flush=True)
        rows.append(dict(corpus=name, n_draws=len(d),
                         fixed_mean=fixed.mean(),
                         fixed_sd=fixed.std(ddof=1),
                         train_only_mean=train.mean(),
                         train_only_sd=train.std(ddof=1),
                         delta_pp=d.mean() * 100, p=p,
                         n_train_only_better=int((d > 0).sum()),
                         n_fixed_better=int((d < 0).sum()), n_tied=int((d == 0).sum()),
                         n_draws_seven_equal_7core=same,
                         most_common_seven=common[0], most_common_count=common[1],
                         fixed_scores=";".join("%.4f" % v for v in fixed),
                         train_only_scores=";".join("%.4f" % v for v in train),
                         train_only_sevens=";".join(sets),
                         fixed_budgets=";".join(map(str, fixed_bud)),
                         train_only_budgets=";".join(map(str, bud))))
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(out.p.values)
    out.to_csv(OUT, index=False)
    print(out[["corpus", "delta_pp", "p", "holm_p"]].to_string(index=False))
    print("%.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
