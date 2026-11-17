import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import PAPER_7, TABLE_DIR
import common as R

T = REV = TABLE_DIR

CODES = {"nl": "Dutch", "en": "English", "zh": "Mandarin", "de": "German"}
MARGIN = 3.0
ALPHA = 0.05
B, SEED = 5000, 0


def scores(cell):
    return np.array([float(x) for x in str(cell).split(";")])


def ratios(parts):
    out, te, tr = {}, np.zeros(20), np.zeros(20)
    for c, (_, _, spk) in parts.items():
        rs = []
        for r in range(20):
            u = np.unique(spk)
            np.random.RandomState(1000 + r).shuffle(u)
            train = np.isin(spk, u[:max(1, int(len(u) * 0.8))])
            rs.append((~train).sum() / train.sum())
            te[r] += (~train).sum()
            tr[r] += train.sum()
        out[CODES[c]] = float(np.mean(rs))
    out["pooled"] = float(np.mean(te / tr))
    return out


def tests(d, rho):
    J, m, s2 = len(d), d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + rho) * s2)
    t975, t95 = stats.t.ppf(0.975, J - 1), stats.t.ppf(0.95, J - 1)
    return dict(diff_pp=m, sd_pp=np.sqrt(s2),
                p_wilcoxon=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
                p_corrected=2 * stats.t.sf(abs(m) / se, J - 1),
                lo95=m - t975 * se, hi95=m + t975 * se,
                lo90=m - t95 * se, hi90=m + t95 * se,
                p_tost=max(stats.t.sf((m + MARGIN) / se, J - 1),
                           stats.t.cdf((m - MARGIN) / se, J - 1)))


def family(name, rows):
    for key in ("p_wilcoxon", "p_corrected"):
        ps = [r[key] for r in rows]
        adj = multipletests(ps, method="holm")[1] if len(ps) > 1 else ps
        for r, a in zip(rows, adj):
            r[key + "_holm"] = a
    for r in rows:
        r["family"] = name
    return rows


def main():
    parts = {c: R.load("7 Core Prosodic", None, None, c) for c in CODES}
    rho = ratios(parts)
    print("test/train utterance ratio:", {k: round(v, 3) for k, v in rho.items()})

    claims = []
    pc = pd.read_csv(REV / "pooled.csv")
    g = pc[pc.tier == "WavLM-Large MSP"].sort_values("draw")
    d = 100 * (g.raw.values - g.supcon.values)
    claims += family("svm_vs_joint", [dict(comparison="SVM without SupCon - joint model",
                                           corpus="pooled", higher=int((d > 0).sum()),
                                           **tests(d, rho["pooled"]))])
    t6 = pd.read_csv(T / "joint.csv").sort_values("si_f1", ascending=False)
    s6 = {r.feature_set: scores(r.si_scores) for r in t6.itertuples()}
    names = list(t6.feature_set)
    claims += family("joint_neighbours", [
        dict(comparison="%s - %s" % (a, b), corpus="pooled",
             **tests(100 * (s6[a] - s6[b]), rho["pooled"]))
        for a, b in zip(names, names[1:])])
    nb = pd.read_csv(T / "monolingual.csv")

    def mono(corp, tier):
        return scores(nb[(nb.corpus == corp) & (nb.tier == tier)].scores.iloc[0])

    claims += family("size_matched", [
        dict(comparison="WavLM-Large MSP - XLSR-53", corpus=c,
             **tests(100 * (mono(c, "WavLM-Large MSP") - mono(c, "XLSR-53")), rho[c]))
        for c in CODES.values()])
    jv = pd.read_csv(T / "joint_vs_specialists.csv")
    for tier, key in (("WavLM-Large MSP", "pooling_wavlm"), ("7 Core Prosodic", "pooling_prosodic")):
        claims += family(key, [
            dict(comparison="joint - specialist, " + tier, corpus=r.corpus,
                 **tests(100 * (scores(r.joint_scores) - scores(r.mono_scores)), rho[r.corpus]))
            for r in jv[jv.tier == tier].itertuples()])
    pl = pd.read_csv(REV / "per_language.csv")
    for tier in pl.tier.unique():
        w = pl[pl.tier == tier].pivot(index="draw", columns="arm", values="f1")
        d = 100 * (w.perlang.values - w.pooled.values)
        claims += family("perlang_" + tier, [dict(comparison="per-language - pooled scaler, " + tier,
                                                   corpus="pooled", higher=int((d > 0).sum()),
                                                   **tests(d, rho["pooled"]))])
    db = pd.read_csv(T / "dutch_benchmarks.csv").set_index("feature_set").scores
    five = ["Wav2Vec2-Large MSP", "WavLM-Large MSP", "emotion2vec\\_base", "7 Core Prosodic", "eGeMAPS"]
    claims += family("dutch_pairwise", [
        dict(comparison="%s - %s" % (a, b), corpus="Dutch",
             **tests(100 * (scores(db[a]) - scores(db[b])), rho["Dutch"]))
        for i, a in enumerate(five) for b in five[i + 1:]])
    claims += family("xlsr_vs_prosody", [
        dict(comparison="XLSR-53 - 7 Core Prosodic", corpus=c,
             **tests(100 * (mono(c, "XLSR-53") - mono(c, "7 Core Prosodic")), rho[c]))
        for c in CODES.values()])
    ab = pd.read_csv(T / "contrastive_ablation.csv")
    head = []
    for r in ab.itertuples():
        row = dict(comparison="SupCon - raw SVM, " + r.tier, corpus=r.corpus,
                   **tests(scores(r.deltas), rho[r.corpus]))
        row["p_wilcoxon"] = r.wilcoxon_p
        head.append(row)
    claims += family("supcon_head", head)
    ss = pd.read_csv(T / "selection_sensitivity.csv")
    claims += family("train_only_selection", [
        dict(comparison="train-only seven - 7 Core Prosodic", corpus=r.corpus,
             **tests(100 * (scores(r.train_only_scores) - scores(r.fixed_scores)), rho[r.corpus]))
        for r in ss.itertuples()])
    lk = pd.read_csv(T / "leakage.csv")
    leak = []
    for r in lk.itertuples():
        c = next(v for v in CODES.values() if str(r.corpus).startswith(v))
        m, sd, J = 100 * (r.leaked_f1 - r.clean_f1), r.delta_sd_pp, r.n_draws
        pc = lambda m_, s_: 2 * stats.t.sf(abs(m_) / np.sqrt((1 / J + rho[c]) * s_ ** 2), J - 1)
        leak.append(dict(comparison="leaked - clean, " + r.feature_set, corpus=c, diff_pp=m,
                         sd_pp=sd, p_wilcoxon=r.p_wilcoxon, p_corrected=pc(m, sd),
                         p_corrected_bound=pc(abs(m) + 0.01, max(sd - 0.05, 1e-9))))
    claims += family("leakage", leak)
    ct = pd.DataFrame(claims)

    eq = ct[ct.family == "pooling_wavlm"].copy()
    eq["smallest_margin_pp"] = eq[["lo90", "hi90"]].abs().max(axis=1)
    se = (eq.hi95 - eq.lo95) / (2 * stats.t.ppf(0.975, 19))
    eq["smallest_margin_bonferroni_pp"] = eq.diff_pp.abs() + stats.t.ppf(1 - ALPHA / 4, 19) * se
    sd = {r.corpus: (r.joint_sd, r.mono_sd) for r in jv[jv.tier == "WavLM-Large MSP"].itertuples()}
    eq["joint_sd"] = eq.corpus.map(lambda c: sd[c][0])
    eq["mono_sd"] = eq.corpus.map(lambda c: sd[c][1])

    heat = pd.read_csv(T / "effect_sizes_all14.csv")
    heat = heat.set_index(heat.columns[0])
    rng = np.random.default_rng(SEED)

    def cohen_d(x, y):
        a, b = x[y == 1], x[y == 0]
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
        return (a.mean() - b.mean()) / sp

    rows = []
    for c, (X, y, spk) in parts.items():
        groups = np.unique(spk)
        idx = {g_: np.where(spk == g_)[0] for g_ in groups}
        boots = np.empty((B, len(PAPER_7)))
        for b in range(B):
            take = np.concatenate([idx[g_] for g_ in rng.choice(groups, len(groups), replace=True)])
            yb = y[take]
            boots[b] = [cohen_d(X[take, j], yb) if 0 < yb.sum() < len(yb) else np.nan
                        for j in range(len(PAPER_7))]
        n1, n0 = (y == 1).sum(), (y == 0).sum()
        for j, f in enumerate(PAPER_7):
            dj = cohen_d(X[:, j], y)
            se = np.sqrt((n1 + n0) / (n1 * n0) + dj ** 2 / (2 * (n1 + n0)))
            lo, hi = np.nanpercentile(boots[:, j], [2.5, 97.5])
            rows.append(dict(corpus=CODES[c], feature=f, d=dj, figure3=heat.loc[f, CODES[c]],
                             utt_lo=dj - 1.96 * se, utt_hi=dj + 1.96 * se,
                             cluster_lo=lo, cluster_hi=hi, clusters=len(groups)))
    dd = pd.DataFrame(rows)

    REV.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rho]).to_csv(REV / "draw_ratios.csv", index=False)
    ct.to_csv(REV / "corrected_tests.csv", index=False)
    eq.to_csv(REV / "pooling_equivalence.csv", index=False)
    dd.to_csv(REV / "cohen_d_intervals.csv", index=False)
    pd.set_option("display.width", 200)
    print(ct[["family", "comparison", "corpus", "diff_pp", "p_wilcoxon_holm", "p_corrected_holm"]]
          .round(4).to_string(index=False))
    print(eq[["corpus", "diff_pp", "lo95", "hi95", "p_tost", "smallest_margin_pp", "sd_pp"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
