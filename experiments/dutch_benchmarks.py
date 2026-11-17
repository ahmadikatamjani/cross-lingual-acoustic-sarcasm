import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import TABLE_DIR
import dutch_common as U

OUT = TABLE_DIR
BUDGETS = [5, 10, 20, 30, 50, 100]
SIG_TIERS = ["Wav2Vec2-Large MSP", "WavLM-Large MSP", "emotion2vec\\_base",
             "7 Core Prosodic", "eGeMAPS"]


def split(spk, r):
    u = np.unique(spk).copy()
    np.random.RandomState(1000 + r).shuffle(u)
    n_tr = max(1, int(len(u) * 0.8))
    tr_spk = u[:n_tr]
    n_in = max(1, int(n_tr * 0.8))
    tr = np.isin(spk, tr_spk)
    return tr, ~tr, np.isin(spk, tr_spk[:n_in]), np.isin(spk, tr_spk[n_in:])


def prep(X, fit_mask, apply_masks, pca_dim, seed):
    sc = StandardScaler().fit(X[fit_mask])
    out = [sc.transform(X[m]) for m in apply_masks]
    base = sc.transform(X[fit_mask])
    if pca_dim:
        p = PCA(n_components=min(pca_dim, base.shape[0] - 1, base.shape[1]),
                random_state=seed).fit(base)
        base, out = p.transform(base), [p.transform(o) for o in out]
    return base, out


def holm(p):
    o = np.argsort(p)
    m, a, prev = len(p), np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        prev = max(prev, min(1.0, (m - r) * p[i]))
        a[i] = prev
    return a


def main(repeats):
    from scipy.stats import wilcoxon
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, geo, per_draw = [], [], {}
    print("Dutch, %d draws, nested budget from %s" % (repeats, BUDGETS))
    hdr = ("%-22s %6s %8s %16s %16s %9s"
           % ("tier", "dim", "budget", "SupCon F1", "Raw+SVM F1", "delta pp"))
    print()
    print(hdr)
    print("-" * len(hdr))

    for name, key, pca_dim in U.TIERS:
        got = U.load(key)
        if got is None:
            print("%-22s  [unavailable]" % name)
            continue
        X, y, spk = got
        sup, raw, picked = [], [], []
        g_acc = {s: [] for s in ("Raw", "SimCLR", "SupCon")}
        for r in range(repeats):
            tr, te, a, b = split(spk, r)
            if te.sum() < 15 or len(np.unique(y[te])) < 2:
                continue

            best_e, best_s = 30, -1.0
            if b.sum() >= 15 and len(np.unique(y[b])) > 1:
                Xa, (Xb,) = prep(X, a, [b], pca_dim, 1000 + r)
                for e in BUDGETS:
                    m = U.train(Xa, y[a], e, seed=1000 + r)
                    Za, Zb = U.project(m, Xa, Xb)
                    v = U.svm_f1(Za, y[a], Zb, y[b])[1]
                    if v > best_s:
                        best_s, best_e = v, e
            picked.append(best_e)

            Xtr, (Xte,) = prep(X, tr, [te], pca_dim, 1000 + r)
            ytr, yte = y[tr], y[te]
            m = U.train(Xtr, ytr, best_e, seed=1000 + r)
            Ztr, Zte = U.project(m, Xtr, Xte)
            sup.append(U.svm_f1(Ztr, ytr, Zte, yte)[1])
            raw.append(U.svm_f1(Xtr, ytr, Xte, yte)[1])

            if name in U.GEOMETRY:
                lp = LogisticRegression(class_weight="balanced", max_iter=1000)
                ms = U.train(Xtr, ytr, best_e, seed=1000 + r, labelled=False)
                Str, Ste = U.project(ms, Xtr, Xte)
                for space, A_, B_ in (("Raw", Xtr, Xte), ("SimCLR", Str, Ste),
                                      ("SupCon", Ztr, Zte)):
                    g_acc[space].append((
                        float(silhouette_score(B_, yte)), U.geometry(B_, yte),
                        float(f1_score(yte, lp.fit(A_, ytr).predict(B_),
                                       average="macro")),
                        U.svm_f1(A_, ytr, B_, yte)[1]))

        if not sup:
            continue
        S, R = np.array(sup), np.array(raw)
        per_draw[name] = S
        print("%-22s %6d %8d  %.3f +/- %.3f  %.3f +/- %.3f %+9.2f"
              % (name, X.shape[1] if not pca_dim else pca_dim,
                 Counter(picked).most_common(1)[0][0],
                 S.mean(), S.std(ddof=1), R.mean(), R.std(ddof=1),
                 (S.mean() - R.mean()) * 100))
        rows.append(dict(
            feature_set=name, dim=int(pca_dim or X.shape[1]),
            budget_mode=Counter(picked).most_common(1)[0][0],
            budgets=";".join(str(x) for x in picked),
            supcon_mean=round(float(S.mean()), 6),
            supcon_sd=round(float(S.std(ddof=1)), 6),
            raw_mean=round(float(R.mean()), 6),
            raw_sd=round(float(R.std(ddof=1)), 6),
            delta_pp=round(float((S.mean() - R.mean()) * 100), 3),
            n_draws=len(S),
            scores=";".join("%.6f" % v for v in S),
            raw_scores=";".join("%.6f" % v for v in R)))

        for space, vals in g_acc.items():
            if not vals:
                continue
            v = np.array(vals)
            geo.append(dict(feature_set=name, space=space,
                            sil=round(float(v[:, 0].mean()), 3),
                            sil_sd=round(float(v[:, 0].std(ddof=1)), 3),
                            ratio=round(float(v[:, 1].mean()), 3),
                            lp_f1=round(float(v[:, 2].mean()), 6),
                            svm_f1=round(float(v[:, 3].mean()), 6),
                            svm_sd=round(float(v[:, 3].std(ddof=1)), 6),
                            n_draws=len(v)))

    sig = []
    have = [t for t in SIG_TIERS if t in per_draw]
    pairs = [(a, b) for i, a in enumerate(have) for b in have[i + 1:]]
    for a, b in pairs:
        A, B = per_draw[a], per_draw[b]
        n = min(len(A), len(B))
        d = A[:n] - B[:n]
        try:
            p = float(wilcoxon(A[:n], B[:n]).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        sig.append(dict(tier_a=a, tier_b=b,
                        mean_a=round(float(A[:n].mean()), 6),
                        mean_b=round(float(B[:n].mean()), 6),
                        delta_pp=round(float(d.mean() * 100), 3),
                        wilcoxon_p=p, n_draws=n))
    if sig:
        ps = np.array([s["wilcoxon_p"] for s in sig])
        for s, h in zip(sig, holm(ps)):
            s["holm_p"] = round(float(h), 6)
            s["wilcoxon_p"] = round(s["wilcoxon_p"], 6)
        print()
        print("PAIRWISE, paired Wilcoxon over %d draws, Holm across %d comparisons"
              % (rows[0]["n_draws"], len(sig)))
        print("  %-22s %-22s %8s %10s %10s" % ("tier A", "tier B", "delta", "p", "holm"))
        for s in sorted(sig, key=lambda x: x["holm_p"]):
            print("  %-22s %-22s %+7.2f %10.5f %10.5f %s"
                  % (s["tier_a"], s["tier_b"], s["delta_pp"], s["wilcoxon_p"],
                     s["holm_p"], "*" if s["holm_p"] < 0.05 else ""))

    pd.DataFrame(rows).to_csv(OUT / "dutch_benchmarks.csv", index=False)
    pd.DataFrame(geo).to_csv(OUT / "dutch_geometry.csv", index=False)
    pd.DataFrame(sig).to_csv(OUT / "dutch_pairwise.csv", index=False)
    print("%.1f min, written to %s" % ((time.time() - t0) / 60, OUT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    main(ap.parse_args().repeats)
