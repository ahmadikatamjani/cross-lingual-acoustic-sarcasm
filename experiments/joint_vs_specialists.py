import argparse
import sys
import time
from collections import Counter
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

OUT = R.OUT
CODES = ["nl", "en", "zh", "de"]
TIERS = [("WavLM-Large MSP", "wavlm_msp", "wavlm_msp"),
         ("7 Core Prosodic", None, None)]
BUDGETS = [5, 10, 20, 30, 50, 100]


def train_encoder(Xtr, ytr, seed, epochs):
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
    return sc, m, SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(etr, ytr)


def score(sc, m, svm, Xte, yte):
    if len(np.unique(yte)) < 2:
        return float("nan")
    with torch.no_grad():
        e = m(torch.FloatTensor(sc.transform(Xte)).to(R.DEV)).cpu().numpy()
    return float(f1_score(yte, svm.predict(e), average="macro"))


def inner_split(spk, outer_train, seed):
    u = np.unique(spk[outer_train])
    rng = np.random.RandomState(seed)
    u = u.copy()
    rng.shuffle(u)
    cut = max(1, int(round(len(u) * 0.8)))
    a = outer_train & np.isin(spk, u[:cut])
    b = outer_train & np.isin(spk, u[cut:])
    return a, b


def main(repeats):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for tier, subdir, frag in TIERS:
        data = {}
        for c in CODES:
            got = R.load(tier, subdir, frag, c)
            if got is not None:
                data[c] = got
        dims = {c: data[c][0].shape[1] for c in data}
        if len(data) < 2 or len(set(dims.values())) != 1:
            print("skipping %s (corpora %d, dims %s)" % (tier, len(data), dims))
            continue
        print("%s: %d draws, pooled n=%d" % (tier, repeats, sum(len(data[c][1]) for c in data)))

        mono = {c: [] for c in data}
        joint = {c: [] for c in data}
        picked_j, picked_m = [], {c: [] for c in data}

        for r in range(repeats):
            outer = {}
            for c in data:
                X, y, spk = data[c]
                u = np.unique(spk).copy()
                np.random.RandomState(1000 + r).shuffle(u)
                tr = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
                if (~tr).sum() < 15 or len(np.unique(y[~tr])) < 2:
                    continue
                outer[c] = tr
            if len(outer) < 2:
                continue

            inner = {}
            for c in outer:
                a, b = inner_split(data[c][2], outer[c], 2000 + r)
                if b.sum() >= 10 and len(np.unique(data[c][1][b])) > 1:
                    inner[c] = (a, b)

            best_j, best_j_score = 30, -1.0
            if len(inner) >= 2:
                for e in BUDGETS:
                    Xa = np.vstack([data[c][0][inner[c][0]] for c in inner])
                    ya = np.concatenate([data[c][1][inner[c][0]] for c in inner])
                    sc, m, sv = train_encoder(Xa, ya, 2000 + r, e)
                    vals = [score(sc, m, sv, data[c][0][inner[c][1]],
                                  data[c][1][inner[c][1]]) for c in inner]
                    vals = [v for v in vals if v == v]
                    if vals and np.mean(vals) > best_j_score:
                        best_j_score, best_j = float(np.mean(vals)), e
            picked_j.append(best_j)

            best_m = {}
            for c in outer:
                b_e, b_s = 30, -1.0
                if c in inner:
                    a, b = inner[c]
                    for e in BUDGETS:
                        v = R.fit_eval(data[c][0][a], data[c][1][a],
                                       data[c][0][b], data[c][1][b],
                                       2000 + r, epochs=e)
                        if v == v and v > b_s:
                            b_s, b_e = v, e
                best_m[c] = b_e
                picked_m[c].append(b_e)

            Xp = np.vstack([data[c][0][outer[c]] for c in outer])
            yp = np.concatenate([data[c][1][outer[c]] for c in outer])
            jsc, jm, jsv = train_encoder(Xp, yp, 1000 + r, best_j)
            for c in outer:
                X, y, _ = data[c]
                te = ~outer[c]
                vj = score(jsc, jm, jsv, X[te], y[te])
                vm = R.fit_eval(X[outer[c]], y[outer[c]], X[te], y[te],
                                1000 + r, epochs=best_m[c])
                if vj == vj and vm == vm:
                    joint[c].append(vj)
                    mono[c].append(vm)
        for c in CODES:
            if c not in data or not joint[c]:
                continue
            J, M = np.array(joint[c]), np.array(mono[c])
            d = (J - M) * 100
            try:
                p = (wilcoxon(J, M).pvalue if len(J) > 2 and np.any(d != 0)
                     else float("nan"))
            except ValueError:
                p = float("nan")
            print("  %-9s joint %.4f  specialist %.4f  delta %+.2f pp"
                  % (R.CORPORA[c], J.mean(), M.mean(), d.mean()))
            rows.append({
                "tier": tier, "corpus": R.CORPORA[c],
                "joint_budgets": ";".join(str(x) for x in picked_j),
                "mono_budgets": ";".join(str(x) for x in picked_m[c]),
                "joint_budget_mode": Counter(picked_j).most_common(1)[0][0],
                "mono_budget_mode": Counter(picked_m[c]).most_common(1)[0][0],
                "joint_mean": round(float(J.mean()), 6),
                "joint_sd": round(float(J.std(ddof=1)), 6),
                "mono_mean": round(float(M.mean()), 6),
                "mono_sd": round(float(M.std(ddof=1)), 6),
                "delta_pp": round(float(d.mean()), 4),
                "wilcoxon_p": round(float(p), 5) if p == p else None,
                "n_draws": len(J),
                "joint_scores": ";".join("%.6f" % v for v in J),
                "mono_scores": ";".join("%.6f" % v for v in M)})

    if rows:
        pd.DataFrame(rows).to_csv(OUT / "joint_vs_specialists.csv", index=False)
        print("%.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    main(ap.parse_args().repeats)
