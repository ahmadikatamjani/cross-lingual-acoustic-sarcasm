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
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import PAPER_7, TABLE_DIR
import common as R
from zeroshot import grouped_ci, random_baselines
from zeroshot import load as load_prosodic

NAME = {"nl": "Dutch", "en": "English", "zh": "Chinese", "de": "German"}
BLOCKS = ["7 Core Prosodic", "Wav2Vec2-Large MSP", "WavLM-Large MSP"]
ARMS = ("frozen", "target")
EPOCHS = 30
SEEDS = list(range(42, 62))
OUT = TABLE_DIR


def data(block, code):
    if block == "7 Core Prosodic":
        return load_prosodic(code, PAPER_7)
    tier = [t for t in R.TIERS if t[0] == block][0]
    X, y, spk = R.load(*tier, code)
    return X.astype(np.float64), y, np.asarray(spk).astype(str)


def train(Xs, ys, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xs)
    m = R.Enc(Xs.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.FloatTensor(sc.transform(Xs)).to(R.DEV)
    yt = torch.LongTensor(ys).to(R.DEV)
    m.train()
    for _ in range(EPOCHS):
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
    return sc, m, SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(etr, ys)


def predict(sc, m, svc, X, arm):
    Z = sc.transform(X) if arm == "frozen" else StandardScaler().fit(X).transform(X)
    with torch.no_grad():
        e = m(torch.FloatTensor(Z).to(R.DEV)).cpu().numpy()
    return svc.predict(e)


def main():
    t0, rows = time.time(), []
    for block in BLOCKS:
        D = {c: data(block, c) for c in NAME}
        for s in NAME:
            Xs, ys, _ = D[s]
            for seed in SEEDS:
                sc, m, svc = train(Xs, ys, seed)
                for t in NAME:
                    if t == s:
                        continue
                    Xt, yt, gt = D[t]
                    for arm in ARMS:
                        p = predict(sc, m, svc, Xt, arm)
                        rec = dict(block=block, source=NAME[s], target=NAME[t],
                                   seed=seed, arm=arm,
                                   f1=round(f1_score(yt, p, average="macro"), 6),
                                   pred_pos_rate=round(float(p.mean()), 4))
                        if seed == SEEDS[0]:
                            lo, hi, ng = grouped_ci(yt, p, gt)
                            rec.update(ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                                       n_groups=ng)
                        rows.append(rec)
            print("%-19s source %-8s %d seeds  (%.1f min)"
                  % (block, NAME[s], len(SEEDS), (time.time() - t0) / 60), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "zeroshot_standardised_seeds.csv", index=False)

    keys = ["block", "source", "target", "arm"]
    g = long.groupby(keys)
    summ = g.f1.agg(mean_f1="mean", sd_f1="std", min_f1="min", max_f1="max",
                    n_seeds="count").reset_index()
    one = long.assign(one_class=np.maximum(long.pred_pos_rate, 1 - long.pred_pos_rate))
    summ = summ.merge(one.groupby(keys).one_class.mean().rename("mean_one_class_rate")
                      .reset_index(), on=keys)
    s42 = long[long.seed == SEEDS[0]][keys + ["f1", "ci_lo", "ci_hi", "n_groups"]]
    summ = summ.merge(s42.rename(columns={"f1": "seed42_f1"}), on=keys)
    rb = random_baselines()
    code = {v: k for k, v in NAME.items()}
    summ["random_baseline"] = [round(rb[code[t]], 4) for t in summ.target]
    for c in ("mean_f1", "sd_f1", "mean_one_class_rate"):
        summ[c] = summ[c].round(6)
    summ.to_csv(OUT / "zeroshot_standardised.csv", index=False)
    for (block, arm), d in summ.groupby(["block", "arm"], sort=False):
        print("%-19s %-7s mean F1 %.3f  CI above chance %d/12"
              % (block, arm, d.mean_f1.mean(), (d.ci_lo > d.random_baseline).sum()))
    print("%.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
