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
EPOCHS = 30
GRID = ([("rbf", 1.0, "scale")]
        + [("rbf", c, g) for c in (0.01, 0.1, 1.0, 10.0)
           for g in (0.0001, 0.001, 0.01, 0.1)]
        + [("linear", c, "scale") for c in (0.01, 0.1, 1.0, 10.0)])
FIXED = GRID[0]


def encode(Xtr, ytr, Xte, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    m = R.Enc(Ztr.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.FloatTensor(Ztr).to(R.DEV)
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
        return (m(Xt).cpu().numpy(),
                m(torch.FloatTensor(Zte).to(R.DEV)).cpu().numpy())


def grid_f1(Etr, ytr, Ete, yte):
    out = {}
    for k, c, g in GRID:
        sv = SVC(kernel=k, C=c, gamma=g, class_weight="balanced").fit(Etr, ytr)
        out[(k, c, g)] = float(f1_score(yte, sv.predict(Ete), average="macro"))
    return out


def outer_mask(spk, r):
    u = np.unique(spk).copy()
    np.random.RandomState(1000 + r).shuffle(u)
    return np.isin(spk, u[:max(1, int(len(u) * 0.8))])


def inner_mask(spk, tr, r):
    u = np.unique(spk[tr]).copy()
    np.random.RandomState(2000 + r).shuffle(u)
    cut = max(1, int(round(len(u) * 0.8)))
    return tr & np.isin(spk, u[:cut]), tr & np.isin(spk, u[cut:])


def main(repeats, only):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cells, t0 = [], [], time.time()
    for tier, subdir, frag in TIERS:
        for code in CODES:
            if only and code != only:
                continue
            got = R.load(tier, subdir, frag, code)
            if got is None:
                continue
            X, y, spk = got
            fixed, nested, oracle, picks = [], [], [], []
            per_point = {g: [] for g in GRID}
            for r in range(repeats):
                tr = outer_mask(spk, r)
                te = ~tr
                if te.sum() < 15 or len(np.unique(y[te])) < 2:
                    continue
                Etr, Ete = encode(X[tr], y[tr], X[te], 1000 + r)
                sco = grid_f1(Etr, y[tr], Ete, y[te])
                for g in GRID:
                    per_point[g].append(sco[g])
                fixed.append(sco[FIXED])
                oracle.append(max(sco.values()))

                a, b = inner_mask(spk, tr, r)
                pick = FIXED
                if b.sum() >= 10 and len(np.unique(y[b])) > 1:
                    Ea, Eb = encode(X[a], y[a], X[b], 2000 + r)
                    inner = grid_f1(Ea, y[a], Eb, y[b])
                    pick = max(inner, key=lambda k: (inner[k], k == FIXED))
                picks.append(pick)
                nested.append(sco[pick])
            if not fixed:
                continue
            F, N, O = np.array(fixed), np.array(nested), np.array(oracle)
            try:
                p = (wilcoxon(N, F).pvalue if np.any(N != F) else float("nan"))
            except ValueError:
                p = float("nan")
            mode = Counter(picks).most_common(1)[0]
            spread = (max(np.mean(v) for v in per_point.values())
                      - min(np.mean(v) for v in per_point.values())) * 100
            print("%-20s %-9s fixed %.4f  nested %.4f (%+.2f pp)"
                  % (tier, R.CORPORA[code], F.mean(), N.mean(), (N.mean() - F.mean()) * 100))
            rows.append({
                "tier": tier, "corpus": R.CORPORA[code], "n_draws": len(F),
                "fixed_mean": round(float(F.mean()), 6),
                "fixed_sd": round(float(F.std(ddof=1)), 6),
                "nested_mean": round(float(N.mean()), 6),
                "nested_sd": round(float(N.std(ddof=1)), 6),
                "oracle_mean": round(float(O.mean()), 6),
                "nested_delta_pp": round(float((N.mean() - F.mean()) * 100), 3),
                "oracle_delta_pp": round(float((O.mean() - F.mean()) * 100), 3),
                "wilcoxon_p": round(float(p), 5) if p == p else None,
                "grid_spread_pp": round(float(spread), 3),
                "modal_pick": str(mode[0]), "modal_pick_n": mode[1],
                "fixed_scores": ";".join("%.6f" % v for v in F),
                "nested_scores": ";".join("%.6f" % v for v in N)})
            for g, v in per_point.items():
                cells.append({"tier": tier, "corpus": R.CORPORA[code],
                              "kernel": g[0], "C": g[1], "gamma": g[2],
                              "mean_f1": round(float(np.mean(v)), 6),
                              "sd_f1": round(float(np.std(v, ddof=1)), 6)
                              if len(v) > 1 else None, "n": len(v)})
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "svm_grid.csv", index=False)
        pd.DataFrame(cells).to_csv(OUT / "svm_grid_points.csv", index=False)
        print("%.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()
    main(a.repeats, a.corpus)
