import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEATURE_DIR, SEED, TABLE_DIR

N_RANDOM = 10000
CORPORA = [("Dutch", "dutch"), ("MUStARD++", "mustardpp"), ("MCSD", "mcsd"), ("MuSaG", "musag")]


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def decode_labels(raw):
    arr = np.asarray(raw)
    if arr.dtype.kind in "iub":
        return arr.astype(int)
    return (np.char.lower(arr.astype(str)) == "sarcastic").astype(int)


def run_chance_baselines():
    rows = []
    for name, stem in CORPORA:
        path = FEATURE_DIR / "wavlm_msp" / ("%s_wavlm_msp_embeddings.npz" % stem)
        if not path.exists():
            print("missing: %s" % path)
            continue
        y = decode_labels(np.load(path, allow_pickle=True)["labels"])
        pos = float(np.mean(y))
        zeros = np.zeros((len(y), 1))

        maj = DummyClassifier(strategy="most_frequent").fit(zeros, y)
        f1_maj = macro_f1(y, maj.predict(zeros))

        strat = DummyClassifier(strategy="stratified", random_state=np.random.RandomState(SEED))
        strat.fit(zeros, y)
        draws = [macro_f1(y, strat.predict(zeros)) for _ in range(N_RANDOM)]
        f1_strat = float(np.mean(draws))

        rows.append({
            "Corpus": name,
            "N": len(y),
            "Positive rate": round(pos, 3),
            "Majority-class macro-F1": round(f1_maj, 4),
            "Stratified-random macro-F1": round(f1_strat, 4),
            "Random draws": N_RANDOM,
            "Random draw sd": round(float(np.std(draws, ddof=1)), 4),
        })
        print("%-11s N=%-5d pos=%.3f  majority %.4f  random %.4f"
              % (name, len(y), pos, f1_maj, f1_strat))
    return rows


if __name__ == "__main__":
    np.random.seed(SEED)
    rows = run_chance_baselines()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / "chance_baselines.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
