import sys
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
from config import ALL_PROSODIC, PAPER_7, PROSODY_DIR, TABLE_DIR
import common as R

SRC = PROSODY_DIR
NAME = {"nl": "Dutch", "en": "English", "zh": "Chinese", "de": "German"}
CHANCE_CSV = TABLE_DIR / "chance_baselines.csv"
CHANCE_NAME = {"nl": "Dutch", "en": "MUStARD++", "zh": "MCSD", "de": "MuSaG"}
EPOCHS, SEED = 30, 42
NBOOT_IID, NBOOT = 1000, 10000


def random_baselines():
    d = pd.read_csv(CHANCE_CSV).set_index("Corpus")
    return {c: float(d.loc[n, "Stratified-random macro-F1"])
            for c, n in CHANCE_NAME.items()}


def load(code, feats=PAPER_7):
    df = pd.read_csv(SRC / (code + "_prosody.csv")).dropna(subset=feats)
    y = (df["label"].astype(str).str.lower() == "sarcastic").astype(int).values
    return df[feats].values.astype(float), y, df["speaker_id"].astype(str).values


def cohens_d(X, y):
    a, b = X[y == 1], X[y == 0]
    sd = np.sqrt(((len(a) - 1) * a.var(0, ddof=1)
                  + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / np.where(sd == 0, np.nan, sd)


def predict(Xtr, ytr, Xte, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = R.Enc(Xtr.shape[1]).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.FloatTensor(Xtr).to(R.DEV)
    yt = torch.LongTensor(ytr).to(R.DEV)
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
        ete = m(torch.FloatTensor(Xte).to(R.DEV)).cpu().numpy()
    return SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(etr, ytr).predict(ete)


def boot(y, p, rng):
    out = np.empty(NBOOT_IID)
    n = len(y)
    for i in range(NBOOT_IID):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        out[i] = f1_score(yb, p[idx], average="macro") if len(np.unique(yb)) > 1 else np.nan
    return np.nanpercentile(out, [2.5, 97.5])


def macro_f1_counts(c):
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return np.where((tp + fn > 0) & (tn + fp > 0), (pos + neg) / 2, np.nan)


def grouped_ci(y, p, groups, nboot=NBOOT, seed=SEED):
    y, p = np.asarray(y), np.asarray(p)
    keys, inv = np.unique(groups, return_inverse=True)
    per = np.zeros((len(keys), 4))
    for j, hit in enumerate([(y == 1) & (p == 1), (y == 0) & (p == 1),
                             (y == 1) & (p == 0), (y == 0) & (p == 0)]):
        np.add.at(per[:, j], inv, hit.astype(float))
    point = float(macro_f1_counts(per.sum(0)))
    if abs(point - f1_score(y, p, average="macro")) > 1e-9:
        raise SystemExit("count-based macro-F1 disagrees with sklearn")
    pick = np.random.RandomState(seed).randint(0, len(keys), (nboot, len(keys)))
    lo, hi = np.nanpercentile(macro_f1_counts(per[pick].sum(axis=1)), [2.5, 97.5])
    return float(lo), float(hi), len(keys)


def floor(pos):
    return max(0.5 * (2 * pos / (1 + pos)), 0.5 * (2 * (1 - pos) / (2 - pos)))


def blocks():
    top7 = {}
    for c in NAME:
        X, y, _ = load(c, ALL_PROSODIC)
        d = np.nan_to_num(np.abs(cohens_d(X, y)))
        top7[c] = [ALL_PROSODIC[i] for i in np.argsort(-d, kind="stable")[:7]]
    return [("7 Core", lambda s: PAPER_7),
            ("source-only top-7", lambda s: top7[s]),
            ("all-14", lambda s: ALL_PROSODIC)]


def main():
    rb_all = random_baselines()
    rng = np.random.RandomState(20260822)
    rows = []
    hdr = ("%-18s %-8s %-8s %6s %16s %16s %7s %8s %s"
           % ("block", "source", "target", "F1", "grouped 95% CI", "utterance CI",
              "random", "pred_pos", "verdict"))
    print(hdr)
    print("-" * len(hdr))
    for block, feats_for in blocks():
        for s in NAME:
            feats = feats_for(s)
            Xs, ys, _ = load(s, feats)
            for t in NAME:
                if t == s:
                    continue
                Xt, yt, gt = load(t, feats)
                p = predict(Xs, ys, Xt)
                f1 = f1_score(yt, p, average="macro")
                ilo, ihi = boot(yt, p, rng)
                lo, hi, ng = grouped_ci(yt, p, gt)
                fl, rb, pp = floor(yt.mean()), rb_all[t], p.mean()
                if pp < 0.05 or pp > 0.95:
                    v = "collapsed"
                elif lo > rb:
                    v = "ABOVE CHANCE"
                elif pp < 0.25 or pp > 0.75:
                    v = "skewed"
                else:
                    v = "at chance"
                print("%-18s %-8s %-8s %6.3f  [%.3f, %.3f]  [%.3f, %.3f] %7.3f %8.3f %s"
                      % (block, NAME[s], NAME[t], f1, lo, hi, ilo, ihi, rb, pp, v))
                rows.append(dict(block=block, source=NAME[s], target=NAME[t],
                                 f1=round(f1, 6), ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                                 iid_lo=round(ilo, 6), iid_hi=round(ihi, 6),
                                 n_groups=ng, single_class_floor=round(fl, 4),
                                 random_baseline=round(rb, 4),
                                 pred_pos_rate=round(float(pp), 4),
                                 true_pos_rate=round(float(yt.mean()), 4), verdict=v,
                                 features=";".join(feats)))
    df = pd.DataFrame(rows)
    out = TABLE_DIR / "zeroshot_prosodic.csv"
    df.to_csv(out, index=False)
    print(df.groupby("block").verdict.value_counts().to_string())
    print("written " + str(out))


if __name__ == "__main__":
    main()
