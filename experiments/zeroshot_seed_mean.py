import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import TABLE_DIR
import zeroshot as C
import zeroshot_standardised as Z

OUT = TABLE_DIR
NBOOT = 10000
TARGET_BLIND = {b: fn for b, fn in C.blocks() if b != "7 Core"}


def data(block, code, src):
    if block in TARGET_BLIND:
        return C.load(code, TARGET_BLIND[block](src))
    return Z.data(block, code)


def seedmean_ci(y, preds, groups):
    keys, inv = np.unique(groups, return_inverse=True)
    pick = np.random.RandomState(42).randint(0, len(keys), (NBOOT, len(keys)))
    acc = np.zeros(NBOOT)
    for p in preds:
        per = np.zeros((len(keys), 4))
        for j, hit in enumerate([(y == 1) & (p == 1), (y == 0) & (p == 1),
                                 (y == 1) & (p == 0), (y == 0) & (p == 0)]):
            np.add.at(per[:, j], inv, hit.astype(float))
        acc += C.macro_f1_counts(per[pick].sum(axis=1))
    lo, hi = np.nanpercentile(acc / len(preds), [2.5, 97.5])
    return float(lo), float(hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rb = C.random_baselines()
    rows, t0 = [], time.time()
    for block in Z.BLOCKS + list(TARGET_BLIND):
        for s in Z.NAME:
            Xs, ys, _ = data(block, s, s)
            preds = {t: {arm: [] for arm in Z.ARMS} for t in Z.NAME if t != s}
            for seed in Z.SEEDS:
                sc, m, svc = Z.train(Xs, ys, seed)
                for t in preds:
                    Xt, yt, _ = data(block, t, s)
                    for arm in Z.ARMS:
                        preds[t][arm].append(Z.predict(sc, m, svc, Xt, arm))
            for t in preds:
                _, yt, gt = data(block, t, s)
                for arm in Z.ARMS:
                    f = [f1_score(yt, p, average="macro") for p in preds[t][arm]]
                    lo, hi = seedmean_ci(yt, preds[t][arm], gt)
                    rows.append(dict(block=block, source=Z.NAME[s], target=Z.NAME[t], arm=arm,
                                     mean_f1=round(float(np.mean(f)), 6),
                                     sd_f1=round(float(np.std(f, ddof=1)), 6),
                                     ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                                     random_baseline=round(rb[t], 4),
                                     above=bool(lo > rb[t]),
                                     one_class=round(float(np.mean(
                                         [max(p.mean(), 1 - p.mean()) for p in preds[t][arm]])), 6),
                                     n_seeds=len(f)))
            print("%-19s source %-8s done (%.1f min)"
                  % (block, Z.NAME[s], (time.time() - t0) / 60), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "zeroshot_seed_mean.csv", index=False)
    for (b, arm), g in d.groupby(["block", "arm"], sort=False):
        print("%-19s %-6s mean %.3f  intervals above the baseline %2d/12"
              % (b, arm, g.mean_f1.mean(), g.above.sum()))


if __name__ == "__main__":
    main()
