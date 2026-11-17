from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpora"
FEATURE_DIR = DATA_DIR / "features"

RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

DUTCH_DIR = CORPUS_DIR / "dutch"
DUTCH_METADATA = DUTCH_DIR / "dataset.csv"
DUTCH_AUDIO = DUTCH_DIR / "utterance_audio"
MUSTARDPP_DIR = CORPUS_DIR / "mustard_plus_plus"
MCSD_DIR = CORPUS_DIR / "mcsd"
MUSAG_DIR = CORPUS_DIR / "musag"

PROSODY_DIR = FEATURE_DIR / "prosody"
DUTCH_FEATURE_DIR = FEATURE_DIR / "dutch"

SOURCES = {"nl": DUTCH_METADATA, "en": MUSTARDPP_DIR / "mustard++_text.csv",
           "zh": MCSD_DIR / "Files" / "MCSD_v1.xlsx", "de": MUSAG_DIR / "dataset.json"}

XLSR_ID = "facebook/wav2vec2-large-xlsr-53"
XLSR_REVISION = "c3f9d884181a224a6ac87bf8885c84d1cff3384f"

CORPORA = {"nl": "Dutch", "en": "English", "zh": "Mandarin", "de": "German"}

PAPER_7 = ["duration_ms", "energy_slope", "pitch_slope", "intensity_mean_db",
           "pitch_mean_hz", "hnr_db", "shimmer_pct"]

ALL_PROSODIC = ["pitch_mean_hz", "pitch_max_hz", "pitch_min_hz", "pitch_span_st",
                "pitch_std_hz", "pitch_slope", "duration_ms", "intensity_mean_db",
                "hnr_db", "jitter_pct", "shimmer_pct", "spectral_centroid_hz",
                "energy_slope", "pause_ratio"]

SEED = 42
