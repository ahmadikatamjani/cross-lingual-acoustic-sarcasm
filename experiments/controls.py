import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as R
import joint as M
import zeroshot as C
import zeroshot_standardised as Z

OUT = R.OUT
NAME, SEEDS, ARMS = Z.NAME, Z.SEEDS, Z.ARMS
BUDGETS = [5, 10, 20, 30, 50, 100]
NBOOT = 10000
RB = C.random_baselines()
BASE = {NAME[c]: RB[c] for c in NAME}


def fixed_budget_scores():
    ref = pd.read_csv(OUT / "zeroshot_standardised_seeds.csv")
    return {(r.block, r.source, r.target, int(r.seed), r.arm): r.f1 for r in ref.itertuples()}


def train_e(Xs, ys, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xs)
    m = R.Enc(Xs.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.FloatTensor(sc.transform(Xs)).to(R.DEV)
    yt = torch.LongTensor(ys).to(R.DEV)
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
    return sc, m, SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(etr, ys)


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


def summarise(block, s, t, arm, y, g, preds, extra=None):
    f = [f1_score(y, p, average="macro") for p in preds]
    lo, hi = seedmean_ci(y, preds, g)
    row = dict(block=block, source=NAME[s], target=NAME[t], arm=arm,
               mean_f1=float(np.mean(f)), sd_f1=float(np.std(f, ddof=1)), lo=lo, hi=hi,
               above=bool(lo > BASE[NAME[t]]),
               one_class=float(np.mean([max(p.mean(), 1 - p.mean()) for p in preds])))
    row.update(extra or {})
    return row


def part_budget():
    ref = fixed_budget_scores()
    rows, picks, t0 = [], [], time.time()
    for block in Z.BLOCKS:
        D = {c: Z.data(block, c) for c in NAME}
        for s in NAME:
            Xs, ys, gs = D[s]
            preds = {t: {a: [] for a in ARMS} for t in NAME if t != s}
            chosen = {a: [] for a in ARMS}
            for seed in SEEDS:
                u = np.unique(gs).copy()
                np.random.RandomState(seed).shuffle(u)
                itr = np.isin(gs, u[:max(1, int(len(u) * 0.8))])
                ival = ~itr
                best = {a: 30 for a in ARMS}
                if (ival.sum() >= 15 and len(np.unique(ys[ival])) > 1
                        and len(np.unique(ys[itr])) > 1):
                    val = {a: {} for a in ARMS}
                    for e in BUDGETS:
                        sc, m, svc = train_e(Xs[itr], ys[itr], seed, e)
                        for a in ARMS:
                            val[a][e] = f1_score(ys[ival], Z.predict(sc, m, svc, Xs[ival], a),
                                                 average="macro")
                    best = {a: max(BUDGETS, key=lambda e: val[a][e]) for a in ARMS}
                models = {e: train_e(Xs, ys, seed, e) for e in set(best.values())}
                for a in ARMS:
                    chosen[a].append(best[a])
                    picks.append(dict(block=block, source=NAME[s], seed=seed, arm=a, budget=best[a]))
                    sc, m, svc = models[best[a]]
                    for t in preds:
                        preds[t][a].append(Z.predict(sc, m, svc, D[t][0], a))
            for t in preds:
                _, yt, gt = D[t]
                for a in ARMS:
                    fixed = np.mean([ref[(block, NAME[s], NAME[t], sd, a)] for sd in SEEDS])
                    rows.append(summarise(block, s, t, a, yt, gt, preds[t][a],
                                          dict(fixed30_mean=fixed,
                                               budgets=";".join(map(str, chosen[a])))))
            print("budget %-19s source %-8s done (%.1f min)"
                  % (block, NAME[s], (time.time() - t0) / 60), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "budget.csv", index=False)
    pd.DataFrame(picks).to_csv(OUT / "budget_picks.csv", index=False)
    for (b, a), g in d.groupby(["block", "arm"], sort=False):
        print("%-19s %-6s selected %.4f  fixed at 30 epochs %.4f"
              % (b, a, g.mean_f1.mean(), g.fixed30_mean.mean()))


def part_pooled():
    tiers = [("WavLM-Large MSP", "wavlm_msp", "wavlm_msp"), ("7 Core Prosodic", None, None)]
    codes = ["nl", "en", "zh", "de"]
    rows = []
    for tier, sub, frag in tiers:
        parts = {c: R.load(tier, sub, frag, c) for c in codes}
        for r in range(M.DRAWS):
            seed = 1000 + r
            trX, trY, teX, teY, lang = [], [], [], [], []
            for c in codes:
                Xc, yc, spk = parts[c]
                u = np.unique(spk)
                np.random.RandomState(seed).shuffle(u)
                mask = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
                trX.append(Xc[mask])
                trY.append(yc[mask])
                teX.append(Xc[~mask])
                teY.append(yc[~mask])
                lang += [c] * int((~mask).sum())
            Xtr, ytr = np.vstack(trX).astype(np.float64), np.concatenate(trY)
            Xte, yte = np.vstack(teX).astype(np.float64), np.concatenate(teY)
            lang = np.array(lang)
            sup = M.fit_eval(Xtr, ytr, Xte, yte, 100, 1e-4, seed)
            sc = StandardScaler().fit(Xtr)
            p = SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(sc.transform(Xtr), ytr) \
                .predict(sc.transform(Xte))
            row = dict(tier=tier, draw=r, supcon=sup, raw=float(f1_score(yte, p, average="macro")))
            for c in codes:
                k = lang == c
                row["raw_" + c] = float(f1_score(yte[k], p[k], average="macro"))
            rows.append(row)
        g = pd.DataFrame([x for x in rows if x["tier"] == tier])
        d = (g.supcon - g.raw) * 100
        p = wilcoxon(g.supcon, g.raw).pvalue if np.any(d != 0) else float("nan")
        print("%-16s SupCon %.4f  SVM only %.4f  difference %+.2f pp, higher in %d of %d draws"
              % (tier, g.supcon.mean(), g.raw.mean(), d.mean(), (d > 0).sum(), len(d)))
    pd.DataFrame(rows).to_csv(OUT / "pooled.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"budget": part_budget, "pooled": part_pooled}[sys.argv[1]]()
