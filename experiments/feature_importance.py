import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DUTCH_FEATURE_DIR, TABLE_DIR

FEATURES = (["pitch_mean_hz", "pitch_max_hz", "pitch_min_hz", "pitch_span_st",
             "duration_ms", "intensity_mean_db", "hnr_db", "pitch_std_hz",
             "pitch_slope", "jitter_pct", "shimmer_pct", "spectral_centroid_hz",
             "energy_slope", "pause_ratio"]
            + ["mfcc%02d_%s" % (i, stat) for i in range(13) for stat in ("mean", "std")])


def main():
    df = pd.read_csv(DUTCH_FEATURE_DIR / "dutch_40_features.csv").dropna()
    df = df[df["label"].isin(["sincere", "sarcastic"])]
    df["y"] = df["label"].map({"sincere": 0, "sarcastic": 1})
    speakers = sorted(df["participant_id"].unique())

    importances = []
    for spk in speakers:
        train, test = df["participant_id"] != spk, df["participant_id"] == spk
        y_train, y_test = df.loc[train, "y"].values, df.loc[test, "y"].values
        if len(np.unique(y_test)) < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df.loc[train, FEATURES].values.astype(float))
        X_test = scaler.transform(df.loc[test, FEATURES].values.astype(float))
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances.append(result.importances)

    matrix = np.concatenate(importances, axis=1)
    summary = pd.DataFrame({
        "feature": FEATURES,
        "median_importance": np.median(matrix, axis=1),
        "mean_importance": matrix.mean(axis=1),
        "std_importance": matrix.std(axis=1),
    }).sort_values(["median_importance", "mean_importance"], ascending=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "feature_importance.csv", index=False)
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
