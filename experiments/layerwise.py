import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from config import (DUTCH_AUDIO, FEATURE_DIR, MCSD_DIR, MUSAG_DIR, MUSTARDPP_DIR, TABLE_DIR,
                    XLSR_ID, XLSR_REVISION)
import common as R

EMB = FEATURE_DIR / "layerwise"
REV = TABLE_DIR
SR = 16000
ENCODERS = [("XLSR-53", "xlsr", "xlsr"),
            ("Wav2Vec2-Large MSP", "wav2vec2_msp", "wav2vec2_msp"),
            ("WavLM-Large MSP", "wavlm_msp", "wavlm_msp")]
E2V = ("emotion2vec", "emotion2vec", "emotion2vec")
ALL = ENCODERS + [E2V]
DRAWS, MIN_TEST = 20, 15
TOL = 1e-2


def emb_path(code, frag):
    return EMB / ("%s_%s_layers.npy" % (code, frag))


def audio_paths(code):
    if code == "nl":
        fn = np.load(FEATURE_DIR / "xlsr" / "nl_xlsr_embeddings.npz", allow_pickle=True)["filenames"]
        return [DUTCH_AUDIO / str(f) for f in fn]
    stored = next((FEATURE_DIR / "wavlm_msp").glob("%s*wavlm_msp*.npz" % R.STEM[code]))
    keys = np.load(stored, allow_pickle=True)["keys"]
    if code == "en":
        return [MUSTARDPP_DIR / "final_utterance_videos" / ("%s.mp4" % k) for k in keys]
    if code == "zh":
        return [MCSD_DIR / "clips" / ("%s.wav" % k) for k in keys]
    return [MUSAG_DIR / "segments" / ("%s.wav" % k) for k in keys]


def via_ffmpeg(p):
    import librosa
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", str(SR), name],
                       check=True, capture_output=True)
        return librosa.load(name, sr=SR, mono=True)[0]
    finally:
        os.unlink(name)


def models(dev):
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from transformers import (AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor,
                              Wav2Vec2Model, WavLMModel)
    out = {}
    xlsr = snapshot_download(XLSR_ID, revision=XLSR_REVISION)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(xlsr)
    out["XLSR-53"] = (
        lambda w: {"input_values": fe(w, sampling_rate=SR, return_tensors="pt").input_values},
        Wav2Vec2Model.from_pretrained(xlsr).to(dev).eval())
    mid = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    pe = AutoFeatureExtractor.from_pretrained(mid)
    out["Wav2Vec2-Large MSP"] = (
        lambda w: dict(pe(w, sampling_rate=SR, return_tensors="pt", padding=True)),
        AutoModel.from_pretrained(mid).to(dev).eval())
    path = hf_hub_download("tiantiaf/wavlm-large-msp-podcast-emotion-dim", "model.safetensors")
    state = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("backbone_model."):
                state[k[len("backbone_model."):]] = f.get_tensor(k)
    base = snapshot_download("microsoft/wavlm-large")
    wl = WavLMModel.from_pretrained(base)
    wl.load_state_dict(state, strict=False)
    we = Wav2Vec2FeatureExtractor.from_pretrained(base)
    out["WavLM-Large MSP"] = (
        lambda w: dict(we(w, sampling_rate=SR, return_tensors="pt", padding=True)),
        wl.to(dev).eval())
    return out


def hidden_means(prep, model, w, dev):
    import torch
    inp = {k: v.to(dev) for k, v in prep(w).items()}
    with torch.no_grad():
        hs = model(**inp, output_hidden_states=True).hidden_states
    return np.stack([h.squeeze(0).mean(dim=0).float().cpu().numpy() for h in hs])


def extract():
    import librosa
    import torch
    EMB.mkdir(parents=True, exist_ok=True)
    REV.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ms = models(dev)
    t0 = time.time()
    for code in R.CORPORA:
        if all(emb_path(code, frag).exists() for _, _, frag in ENCODERS):
            print("%s: already extracted" % code, flush=True)
            continue
        paths = audio_paths(code)
        n = len(paths)
        ref = {name: R.load(name, sub, frag, code)[0] for name, sub, frag in ENCODERS}
        assert all(len(v) == n for v in ref.values()), "row counts differ for %s" % code
        out = {name: np.zeros((n, m.config.num_hidden_layers + 1, m.config.hidden_size),
                              np.float32) for name, (_, m) in ms.items()}
        for i, p in enumerate(paths):
            w = librosa.load(str(p), sr=SR, mono=True)[0]
            w_ff = via_ffmpeg(p) if p.suffix.lower() == ".mp4" else w
            for name, (prep, m) in ms.items():
                x = w_ff if name == "XLSR-53" else w
                try:
                    out[name][i] = hidden_means(prep, m, x, dev)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    out[name][i] = hidden_means(prep, m.to("cpu"), x, "cpu")
                    m.to(dev)
            if (i + 1) % 200 == 0 or i + 1 == n:
                print("  %s %d/%d  %.1f min" % (code, i + 1, n, (time.time() - t0) / 60),
                      flush=True)
        for name, sub, frag in ENCODERS:
            diff = np.abs(out[name][:, -1, :] - ref[name]).max(axis=1)
            if (diff > TOL).any():
                raise SystemExit("%s %s: the last layer differs from the stored embeddings"
                                 % (name, code))
        for name, sub, frag in ENCODERS:
            np.save(emb_path(code, frag), out[name])
    print("extraction done in %.1f min" % ((time.time() - t0) / 60))


def extract_e2v():
    import torch
    import torch.nn.functional as F
    from funasr import AutoModel
    from funasr.utils.load_utils import load_audio_text_image_video
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModel(model="iic/emotion2vec_base", disable_update=True, device=dev).model.eval()
    extra = m.modality_encoders["AUDIO"].modality_cfg.num_extra_tokens
    cap = []
    m.blocks[0].register_forward_pre_hook(lambda mod, args: cap.append(args[0]))
    for b in m.blocks:
        b.register_forward_hook(lambda mod, args, out: cap.append(out[0]))
    name, sub, frag = E2V
    t0 = time.time()
    for code in R.CORPORA:
        if emb_path(code, frag).exists():
            print("%s: already extracted" % code, flush=True)
            continue
        paths = audio_paths(code)
        ref = R.load(name, sub, frag, code)[0]
        assert len(ref) == len(paths), "row counts differ for %s" % code
        out = np.zeros((len(paths), len(m.blocks) + 1, ref.shape[1]), np.float32)
        for i, p in enumerate(paths):
            cap.clear()
            wav = load_audio_text_image_video([str(p)], fs=SR, audio_fs=SR, data_type="sound")[0]
            src = wav.to(dev)
            if m.cfg.normalize:
                src = F.layer_norm(src, src.shape)
            with torch.no_grad():
                m.extract_features(src.view(1, -1), padding_mask=None)
            out[i] = np.stack([h[:, extra:].squeeze(0).mean(0).float().cpu().numpy() for h in cap])
            if (i + 1) % 500 == 0 or i + 1 == len(paths):
                print("  %s %d/%d  %.1f min" % (code, i + 1, len(paths), (time.time() - t0) / 60),
                      flush=True)
        diff = np.abs(out[:, -1, :] - ref).max(axis=1)
        if (diff > TOL).any():
            raise SystemExit("emotion2vec %s: the last layer differs from the stored embeddings" % code)
        np.save(emb_path(code, frag), out)
    print("extraction done in %.1f min" % ((time.time() - t0) / 60))


def draws(y, spk):
    u0, out = np.unique(spk), []
    for r in range(DRAWS):
        u = u0.copy()
        np.random.RandomState(1000 + r).shuffle(u)
        n_tr = max(1, int(len(u) * 0.8))
        tr_spk, te_spk = u[:n_tr], u[n_tr:]
        tr, te = np.isin(spk, tr_spk), np.isin(spk, te_spk)
        if te.sum() < MIN_TEST or len(np.unique(y[te])) < 2:
            continue
        n_in = max(1, int(n_tr * 0.8))
        itr, ival = np.isin(spk, tr_spk[:n_in]), np.isin(spk, tr_spk[n_in:])
        ok = ival.sum() >= MIN_TEST and len(np.unique(y[ival])) > 1 and len(np.unique(y[itr])) > 1
        out.append((r, tr, te, itr if ok else None, ival if ok else None))
    return out


def svm_f1(Xa, ya, Xb, yb):
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler().fit(Xa)
    pred = SVC(kernel="rbf", class_weight="balanced", C=1.0).fit(sc.transform(Xa), ya)
    return float(f1_score(yb, pred.predict(sc.transform(Xb)), average="macro"))


def layer_task(path, layer, y, spk, encoder, corpus):
    A = np.load(path, mmap_mode="r")
    X = (A.mean(axis=1) if layer == "avg" else A[:, layer, :]).astype(np.float64)
    rows = []
    for r, tr, te, itr, ival in draws(y, spk):
        rows.append(dict(encoder=encoder, corpus=corpus, layer=layer, draw=r,
                         test_f1=svm_f1(X[tr], y[tr], X[te], y[te]),
                         val_f1=(svm_f1(X[itr], y[itr], X[ival], y[ival])
                                 if itr is not None else np.nan)))
    return rows


def sweep(jobs, only=None):
    from joblib import Parallel, delayed
    encs = [e for e in ALL if only is None or e[0] == only]
    tasks = []
    for code in R.CORPORA:
        for name, sub, frag in encs:
            p = emb_path(code, frag)
            _, y, spk = R.load(name, sub, frag, code)
            L = np.load(p, mmap_mode="r").shape[1]
            tasks += [(str(p), l, y, spk, name, R.CORPORA[code]) for l in list(range(L)) + ["avg"]]
    print("%d tasks (layer x encoder x corpus), %d draws each, %d jobs" % (len(tasks), DRAWS, jobs),
          flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(layer_task)(*t) for t in tasks)
    df = pd.DataFrame([r for rows in res for r in rows])
    out = REV / "layerwise_svm_draws.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.encoder.isin([e[0] for e in encs])], df], ignore_index=True)
    df.to_csv(out, index=False)
    print("sweep done in %.1f min -> layerwise_svm_draws.csv" % ((time.time() - t0) / 60))


def corrected(d, rho):
    from scipy import stats
    J, m, s2 = len(d), d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + rho) * s2)
    return m, (2 * stats.t.sf(abs(m) / se, J - 1) if se > 0 else 1.0)


def report():
    df = pd.read_csv(REV / "layerwise_svm_draws.csv")
    rho = pd.read_csv(REV / "draw_ratios.csv").iloc[0].to_dict()
    rows, curves = [], []
    for (enc, corpus), g in df.groupby(["encoder", "corpus"], sort=False):
        single = g[g.layer != "avg"].assign(layer=lambda t: t.layer.astype(int))
        L = single.layer.max()
        test = single.pivot(index="draw", columns="layer", values="test_f1")
        val = single.pivot(index="draw", columns="layer", values="val_f1")
        avg = g[g.layer == "avg"].set_index("draw").test_f1.reindex(test.index)
        last = test[L]
        pick = val.idxmax(axis=1).where(val.notna().any(axis=1), L).astype(int)
        chosen = pd.Series([test.loc[r, pick[r]] for r in test.index], index=test.index)
        mean_by_layer = test.mean()
        best = int(mean_by_layer.idxmax())
        d_ch, p_ch = corrected((chosen - last).values * 100, rho[corpus])
        d_av, p_av = corrected((avg - last).values * 100, rho[corpus])
        rows.append(dict(encoder=enc, corpus=corpus, n_draws=len(test), layers=L + 1,
                         last=last.mean(), chosen=chosen.mean(), avg=avg.mean(),
                         best_fixed_layer=best, best_fixed=mean_by_layer[best],
                         chosen_minus_last_pp=d_ch, p_chosen=p_ch,
                         avg_minus_last_pp=d_av, p_avg=p_av,
                         layers_chosen=";".join("%d:%d" % (k, v) for k, v in
                                                sorted(pick.value_counts().items()))))
        for l, s in test.items():
            curves.append(dict(encoder=enc, corpus=corpus, layer=l, depth=l / L,
                               mean_f1=s.mean(), sd_f1=s.std(ddof=1)))
    out = pd.DataFrame(rows)
    out.to_csv(REV / "layerwise_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(REV / "layerwise_curve.csv", index=False)
    print(out.round(3).to_string(index=False))


def chosen_layers(name, corpus, last):
    sw = pd.read_csv(REV / "layerwise_svm_draws.csv")
    g = sw[(sw.encoder == name) & (sw.corpus == corpus) & (sw.layer != "avg")]
    val = g.assign(layer=g.layer.astype(int)).pivot(index="draw", columns="layer",
                                                     values="val_f1")
    return {r: (int(v.idxmax()) if v.notna().any() else last) for r, v in val.iterrows()}


def pipeline():
    import monolingual as N
    out = REV / "layerwise_pipeline_draws.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["encoder", "corpus"])
    t0 = time.time()
    for code in R.CORPORA:
        for name, sub, frag in ALL:
            corpus = R.CORPORA[code]
            if ((done.encoder == name) & (done.corpus == corpus)).any():
                continue
            A = np.load(emb_path(code, frag), mmap_mode="r")
            _, y, spk = R.load(name, sub, frag, code)
            pick = chosen_layers(name, corpus, A.shape[1] - 1)
            rows = []
            for r, tr, te, itr, ival in draws(y, spk):
                X = np.ascontiguousarray(A[:, pick[r], :], dtype=np.float64)
                seed, best = 1000 + r, N.BUDGETS[len(N.BUDGETS) // 2]
                if itr is not None:
                    val = {}
                    for e in N.BUDGETS:
                        v = R.fit_eval(X[itr], y[itr], X[ival], y[ival], seed, epochs=e)
                        if v == v:
                            val[e] = v
                    best = max(val, key=val.get) if val else best
                rows.append(dict(encoder=name, corpus=corpus, draw=r, layer=pick[r],
                                 last=A.shape[1] - 1, budget=best,
                                 test_f1=R.fit_eval(X[tr], y[tr], X[te], y[te], seed,
                                                    epochs=best)))
            cell = pd.DataFrame(rows)
            cell.to_csv(out, mode="a", header=not out.exists(), index=False)
            print("%-20s %-9s %.3f  %.1f min" % (name, corpus, cell.test_f1.mean(),
                                                 (time.time() - t0) / 60), flush=True)


def compare():
    from statsmodels.stats.multitest import multipletests
    df = pd.read_csv(REV / "layerwise_pipeline_draws.csv")
    rho = pd.read_csv(REV / "draw_ratios.csv").iloc[0].to_dict()
    nb = pd.read_csv(REV / "monolingual.csv")
    corpora = list(R.CORPORA.values())

    def last_layer(corpus, enc):
        s = nb[(nb.corpus == corpus) & (nb.tier == enc)].scores.iloc[0]
        return np.array([float(x) for x in str(s).split(";")])

    def chosen(corpus, enc):
        return df[(df.corpus == corpus) & (df.encoder == enc)].sort_values("draw")

    tests = []

    def family(name, diffs):
        ps = [corrected(d, rho[c])[1] for c, d in diffs]
        holm = multipletests(ps, method="holm")[1]
        for (c, d), p, h in zip(diffs, ps, holm):
            tests.append(dict(family=name, corpus=c, diff_pp=d.mean(), p_corrected=p,
                              p_holm=h, higher=int((d > 0).sum()), n=len(d)))

    summary = []
    for enc, _, _ in [e for e in ALL if (df.encoder == e[0]).any()]:
        diffs = []
        for c in corpora:
            g, ll = chosen(c, enc), last_layer(c, enc)
            assert len(g) == len(ll) == DRAWS, (enc, c)
            diffs.append((c, (g.test_f1.values - ll) * 100))
            summary.append(dict(encoder=enc, corpus=c, last_layer=ll.mean(),
                                chosen=g.test_f1.mean(), sd=g.test_f1.std(ddof=1),
                                draws_on_last=int((g.layer == g["last"]).sum()),
                                layers=";".join("%d:%d" % kv for kv in
                                                sorted(g.layer.value_counts().items())),
                                budgets=";".join("%d:%d" % kv for kv in
                                                 sorted(g.budget.value_counts().items()))))
        family("%s chosen - last layer" % enc, diffs)
    pairs = [("WavLM-Large MSP", "XLSR-53"), ("Wav2Vec2-Large MSP", "WavLM-Large MSP"),
             ("Wav2Vec2-Large MSP", "XLSR-53")]
    if (df.encoder == E2V[0]).any():
        pairs += [(E2V[0], "XLSR-53"), (E2V[0], "Wav2Vec2-Large MSP"), (E2V[0], "WavLM-Large MSP")]
    for a, b in pairs:
        family("%s - %s, last layers" % (a, b),
               [(c, (last_layer(c, a) - last_layer(c, b)) * 100) for c in corpora])
        family("%s - %s, chosen layers" % (a, b),
               [(c, (chosen(c, a).test_f1.values - chosen(c, b).test_f1.values) * 100)
                for c in corpora])
    s, t = pd.DataFrame(summary), pd.DataFrame(tests)
    s.to_csv(REV / "layerwise_pipeline_summary.csv", index=False)
    t.to_csv(REV / "layerwise_pipeline_tests.csv", index=False)
    print(s.round(4).to_string(index=False))
    for name, g in t.groupby("family", sort=False):
        print(name + ": " + "  ".join("%s %+.2f (Holm p %.3f)" % (r.corpus, r.diff_pp, r.p_holm)
                                       for r in g.itertuples()))


def source_layer(name, corpus, last):
    sw = pd.read_csv(REV / "layerwise_svm_draws.csv")
    g = sw[(sw.encoder == name) & (sw.corpus == corpus) & (sw.layer != "avg")]
    m = g.assign(layer=g.layer.astype(int)).groupby("layer").val_f1.mean()
    return int(m.idxmax()) if m.notna().any() else last


def transfer():
    from sklearn.metrics import f1_score
    import zeroshot as C
    import zeroshot_standardised as Z
    import zeroshot_seed_mean as S
    rb = C.random_baselines()
    rows, t0 = [], time.time()
    for name, sub, frag in ENCODERS:
        stored = {c: R.load(name, sub, frag, c) for c in Z.NAME}
        A = {c: np.load(emb_path(c, frag), mmap_mode="r") for c in Z.NAME}
        last = A["nl"].shape[1] - 1
        for s in Z.NAME:
            for choice in ("last", "chosen"):
                layer = last if choice == "last" else source_layer(name, R.CORPORA[s], last)
                X = {c: (stored[c][0].astype(np.float64) if choice == "last" else
                         np.ascontiguousarray(A[c][:, layer, :], dtype=np.float64))
                     for c in Z.NAME}
                preds = {t: {arm: [] for arm in Z.ARMS} for t in Z.NAME if t != s}
                for seed in Z.SEEDS:
                    sc, m, svc = Z.train(X[s], stored[s][1], seed)
                    for t in preds:
                        for arm in Z.ARMS:
                            preds[t][arm].append(Z.predict(sc, m, svc, X[t], arm))
                for t in preds:
                    _, yt, gt = stored[t]
                    for arm in Z.ARMS:
                        f = [f1_score(yt, p, average="macro") for p in preds[t][arm]]
                        lo, hi = S.seedmean_ci(yt, preds[t][arm], gt)
                        rows.append(dict(encoder=name, choice=choice, layer=layer,
                                         source=Z.NAME[s], target=Z.NAME[t], arm=arm,
                                         mean_f1=float(np.mean(f)), ci_lo=lo, ci_hi=hi,
                                         random_baseline=rb[t], above=bool(lo > rb[t]),
                                         one_class=float(np.mean([max(p.mean(), 1 - p.mean())
                                                                  for p in preds[t][arm]]))))
                print("%-20s %-6s source %-8s layer %2d  (%.1f min)"
                      % (name, choice, Z.NAME[s], layer, (time.time() - t0) / 60), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(REV / "layerwise_transfer.csv", index=False)
    for (enc, choice, arm), g in d.groupby(["encoder", "choice", "arm"], sort=False):
        print("%-20s %-7s %-7s mean F1 %.3f  above baseline %d/12"
              % (enc, choice, arm, g.mean_f1.mean(), g.above.sum()))


def joint():
    from collections import Counter
    import joint_vs_specialists as J
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    name, sub, frag = ENCODERS[2]
    codes = list(R.CORPORA)
    stored = {c: R.load(name, sub, frag, c) for c in codes}
    A = {c: np.load(emb_path(c, frag), mmap_mode="r") for c in codes}
    last = A["nl"].shape[1] - 1
    sw = pd.read_csv(REV / "layerwise_svm_draws.csv")
    sw = sw[(sw.encoder == name) & (sw.layer != "avg")]
    val = sw.assign(layer=sw.layer.astype(int)).pivot_table(index=["draw", "layer"],
                                                            columns="corpus", values="val_f1")
    joint_layer = {r: int(g.droplevel(0).mean(axis=1).idxmax()) for r, g in val.groupby(level=0)}
    spec_layer = {c: chosen_layers(name, R.CORPORA[c], last) for c in codes}
    y = {c: stored[c][1] for c in codes}
    spk = {c: stored[c][2] for c in codes}
    rows, t0 = [], time.time()

    def feats(c, layer):
        return np.ascontiguousarray(A[c][:, layer, :], dtype=np.float64)

    for choice, todo in (("chosen", range(DRAWS)),):
        for r in todo:
            lj = joint_layer[r]
            Xj = {c: feats(c, lj) for c in codes}
            Xm = {c: feats(c, spec_layer[c][r]) for c in codes}
            outer = {}
            for c in codes:
                u = np.unique(spk[c]).copy()
                np.random.RandomState(1000 + r).shuffle(u)
                tr = np.isin(spk[c], u[:max(1, int(len(u) * 0.8))])
                if (~tr).sum() >= 15 and len(np.unique(y[c][~tr])) >= 2:
                    outer[c] = tr
            inner = {}
            for c in outer:
                a, b = J.inner_split(spk[c], outer[c], 2000 + r)
                if b.sum() >= 10 and len(np.unique(y[c][b])) > 1:
                    inner[c] = (a, b)
            best_j, best_s = 30, -1.0
            if len(inner) >= 2:
                for e in J.BUDGETS:
                    Xa = np.vstack([Xj[c][inner[c][0]] for c in inner])
                    ya = np.concatenate([y[c][inner[c][0]] for c in inner])
                    sc, m, sv = J.train_encoder(Xa, ya, 2000 + r, e)
                    vals = [J.score(sc, m, sv, Xj[c][inner[c][1]], y[c][inner[c][1]])
                            for c in inner]
                    vals = [v for v in vals if v == v]
                    if vals and np.mean(vals) > best_s:
                        best_s, best_j = float(np.mean(vals)), e
            best_m = {}
            for c in outer:
                b_e, b_s = 30, -1.0
                if c in inner:
                    a, b = inner[c]
                    for e in J.BUDGETS:
                        v = R.fit_eval(Xm[c][a], y[c][a], Xm[c][b], y[c][b], 2000 + r, epochs=e)
                        if v == v and v > b_s:
                            b_s, b_e = v, e
                best_m[c] = b_e
            Xp = np.vstack([Xj[c][outer[c]] for c in outer])
            yp = np.concatenate([y[c][outer[c]] for c in outer])
            jsc, jm, jsv = J.train_encoder(Xp, yp, 1000 + r, best_j)
            for c in outer:
                te = ~outer[c]
                rows.append(dict(choice=choice, draw=r, corpus=R.CORPORA[c], joint_layer=lj,
                                 spec_layer=spec_layer[c][r],
                                 joint_budget=best_j, spec_budget=best_m[c],
                                 joint=J.score(jsc, jm, jsv, Xj[c][te], y[c][te]),
                                 spec=R.fit_eval(Xm[c][outer[c]], y[c][outer[c]], Xm[c][te],
                                                 y[c][te], 1000 + r, epochs=best_m[c])))
            print("draw %2d  joint layer %2d budget %3d  (%.1f min)"
                  % (r, lj, best_j, (time.time() - t0) / 60), flush=True)
    d = pd.DataFrame(rows)
    ref = pd.read_csv(REV / "joint_vs_specialists.csv")
    ref = ref[ref.tier == name].set_index("corpus")
    d.to_csv(REV / "layerwise_joint_draws.csv", index=False)
    rho = pd.read_csv(REV / "draw_ratios.csv").iloc[0].to_dict()
    ch = d[d.choice == "chosen"]
    out = []
    for c in R.CORPORA.values():
        g = ch[ch.corpus == c].sort_values("draw")
        js = np.array([float(x) for x in ref.loc[c, "joint_scores"].split(";")])
        ms = np.array([float(x) for x in ref.loc[c, "mono_scores"].split(";")])
        for label, jj, mm in (("last layer", js, ms),
                              ("chosen layers", g.joint.values, g.spec.values)):
            diff = (jj - mm) * 100
            J_, mu, s2 = len(diff), diff.mean(), diff.var(ddof=1)
            se = np.sqrt((1 / J_ + rho[c]) * s2)
            out.append(dict(run=label, corpus=c, joint=jj.mean(), specialist=mm.mean(),
                            loss_pp=-mu, p_corrected=2 * stats.t.sf(abs(mu) / se, J_ - 1),
                            margin_bonferroni_pp=abs(mu) + stats.t.ppf(1 - 0.05 / 4, J_ - 1) * se))
        dj = (g.joint.values - js) * 100
        mu, se = dj.mean(), np.sqrt((1 / len(dj) + rho[c]) * dj.var(ddof=1))
        out.append(dict(run="joint: chosen minus last", corpus=c, joint=g.joint.mean(),
                        specialist=js.mean(), loss_pp=-mu,
                        p_corrected=2 * stats.t.sf(abs(mu) / se, len(dj) - 1),
                        margin_bonferroni_pp=np.nan))
    o = pd.DataFrame(out)
    for run, g in o.groupby("run"):
        o.loc[g.index, "p_holm"] = multipletests(g.p_corrected, method="holm")[1]
    o.to_csv(REV / "layerwise_joint_summary.csv", index=False)
    print(o.round(4).to_string(index=False))
    print("joint layer per draw:", sorted(Counter(joint_layer.values()).items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "extract-e2v", "sweep", "report", "pipeline",
                                     "compare", "transfer", "joint"])
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", help="sweep one encoder and merge it into the CSV")
    a = ap.parse_args()
    {"extract": extract, "extract-e2v": extract_e2v, "sweep": lambda: sweep(a.jobs, a.only),
     "report": report, "pipeline": pipeline, "compare": compare, "transfer": transfer,
     "joint": joint}[a.step]()
