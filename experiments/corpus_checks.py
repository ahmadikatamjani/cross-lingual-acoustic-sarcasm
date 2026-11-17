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
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import PAPER_7, PROSODY_DIR
import common as R
import joint as M

OUT = R.OUT
CODES = ["nl", "en", "zh", "de"]


def mf1(y, p):
    return float(f1_score(y, p, average="macro"))


def fit_pred(Xtr, ytr, tests, epochs, wd, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    m = R.Enc(Xtr.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    Xt = torch.FloatTensor(sc.transform(Xtr)).to(R.DEV)
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
        zte = [m(torch.FloatTensor(sc.transform(X)).to(R.DEV)).cpu().numpy() for X in tests]
    clf = SVC(kernel="rbf", C=1.0, class_weight="balanced").fit(ztr, ytr)
    return [clf.predict(z) for z in zte]


def per_language_standardisation():
    tiers = [M.TIERS[i] for i in (0, 5, 1, 3, 2, 4)]
    rows, t0 = [], time.time()
    for tier, sub, frag in tiers:
        parts = {c: R.load(tier, sub, frag, c) for c in CODES}
        for r in range(M.DRAWS):
            seed = 1000 + r
            tr, te = {}, {}
            for c in CODES:
                Xc, yc, spk = parts[c]
                u = np.unique(spk)
                np.random.RandomState(seed).shuffle(u)
                mask = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
                tr[c] = (Xc[mask].astype(np.float64), yc[mask])
                te[c] = (Xc[~mask].astype(np.float64), yc[~mask])
            ytr = np.concatenate([tr[c][1] for c in CODES])
            yte = np.concatenate([te[c][1] for c in CODES])
            lang = np.concatenate([[c] * len(te[c][1]) for c in CODES])

            pool, = fit_pred(np.vstack([tr[c][0] for c in CODES]), ytr,
                             [np.vstack([te[c][0] for c in CODES])], 100, 1e-4, seed)
            scs = {c: StandardScaler().fit(tr[c][0]) for c in CODES}
            own = np.vstack([scs[c].transform(tr[c][0]) for c in CODES])
            per, trans = fit_pred(own, ytr,
                                  [np.vstack([scs[c].transform(te[c][0]) for c in CODES]),
                                   np.vstack([StandardScaler().fit_transform(te[c][0]) for c in CODES])],
                                  100, 1e-4, seed)
            for arm, p in (("pooled", pool), ("perlang", per), ("perlang_test", trans)):
                row = dict(tier=tier, draw=r, arm=arm, f1=mf1(yte, p))
                for c in CODES:
                    k = lang == c
                    row["f1_" + c] = mf1(yte[k], p[k])
                rows.append(row)
        print("%-18s done (%.1f min)" % (tier, (time.time() - t0) / 60), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_language.csv", index=False)
    for tier in d.tier.unique():
        g = d[d.tier == tier].pivot(index="draw", columns="arm", values="f1")
        diff = (g.perlang - g.pooled) * 100
        p = wilcoxon(g.perlang, g.pooled).pvalue if np.any(diff != 0) else float("nan")
        print("%-18s per-language %.4f vs pooled %.4f  %+.2f pp, higher in %d of %d, p = %.4g"
              % (tier, g.perlang.mean(), g.pooled.mean(), diff.mean(), (diff > 0).sum(), len(diff), p))


def cohen_d(x, y):
    a, b = x[y == 1], x[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def corpus_statistics():
    loaded = {c: R.load("7 Core Prosodic", None, None, c) for c in CODES}

    corr = []
    for c in CODES:
        r = pd.DataFrame(loaded[c][0], columns=PAPER_7).corr()
        for i, a in enumerate(PAPER_7):
            for b in PAPER_7[i + 1:]:
                corr.append(dict(corpus=c, a=a, b=b, r=round(float(r.loc[a, b]), 4)))
    cd = pd.DataFrame(corr)
    cd.to_csv(OUT / "cue_correlations.csv", index=False)
    three = ["duration_ms", "energy_slope", "pitch_slope"]
    t = cd[cd.a.isin(three) & cd.b.isin(three)]
    print("correlations among duration, energy slope and pitch slope: |r| up to %.2f"
          % t.r.abs().max())

    sb = []
    for c in CODES:
        X, y, spk = loaded[c]
        g = pd.DataFrame(dict(spk=spk, y=y)).groupby("spk").y.agg(["size", "mean"])
        both = (g["mean"] > 0) & (g["mean"] < 1)
        purity = float((g["size"] * np.maximum(g["mean"], 1 - g["mean"])).sum() / g["size"].sum())
        f1s = []
        for r in range(20):
            idx = np.random.RandomState(r).permutation(len(y))
            cut = int(len(y) * 0.8)
            tr, te = idx[:cut], idx[cut:]
            glob = int(np.round(y[tr].mean()))
            maj = pd.DataFrame(dict(spk=spk[tr], y=y[tr])).groupby("spk").y.mean()
            pred = np.array([glob if s not in maj.index or maj[s] == 0.5 else int(maj[s] > 0.5)
                             for s in spk[te]])
            f1s.append(mf1(y[te], pred))
        sb.append(dict(corpus=c, groups=len(g), groups_both=int(both.sum()),
                       utt_in_both=round(float(g["size"][both].sum() / g["size"].sum()), 4),
                       majority_share=round(purity, 4), speaker_prior_f1=round(float(np.mean(f1s)), 4),
                       speaker_prior_sd=round(float(np.std(f1s, ddof=1)), 4)))
        print("%s speaker identity alone: macro-F1 %.3f under a random 80/20 split"
              % (c, np.mean(f1s)))
    pd.DataFrame(sb).to_csv(OUT / "speaker_balance.csv", index=False)

    held = []
    for c in CODES:
        X, y, spk = loaded[c]
        u0, sizes = np.unique(spk), []
        for r in range(20):
            u = u0.copy()
            np.random.RandomState(1000 + r).shuffle(u)
            tr = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
            if (~tr).sum() >= 15 and len(np.unique(y[~tr])) > 1:
                sizes.append(int((~tr).sum()))
        held.append(dict(corpus=c, mean=float(np.mean(sizes)), min=min(sizes), max=max(sizes),
                         n_draws=len(sizes)))
    pd.DataFrame(held).to_csv(OUT / "held_out_sizes.csv", index=False)
    print(pd.DataFrame(held).round(1).to_string(index=False))

    nl = pd.read_csv(PROSODY_DIR / "nl_prosody.csv")
    nl["y"] = nl.label.astype(str).str.lower().str.startswith(("sarc", "1")).astype(int)
    typ = nl.utt_id.str.split("_").str[2].map({"d": "declarative", "t": "tag question",
                                                "w": "wh-exclamative"})
    pt = []
    for f in PAPER_7:
        row = dict(feature=f, pooled=round(float(cohen_d(nl[f].values, nl.y.values)), 3))
        for t_ in ("declarative", "tag question", "wh-exclamative"):
            k = (typ == t_).values
            row[t_] = round(float(cohen_d(nl[f].values[k], nl.y.values[k])), 3)
        pt.append(row)
    pd.DataFrame(pt).to_csv(OUT / "dutch_utterance_types.csv", index=False)
    print(pd.DataFrame(pt).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"perlang": per_language_standardisation, "corpus": corpus_statistics}[sys.argv[1]]()
