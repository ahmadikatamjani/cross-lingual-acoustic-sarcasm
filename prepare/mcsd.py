import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from config import MCSD_DIR

EPISODES = MCSD_DIR / "episodes"
CLIPS = MCSD_DIR / "clips"


def download():
    EPISODES.mkdir(parents=True, exist_ok=True)
    for _, r in pd.read_csv(HERE / "mcsd_episodes.csv").iterrows():
        if (EPISODES / ("%s.wav" % r["item"])).exists():
            continue
        subprocess.run(["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
                        "-o", str(EPISODES / ("%s.%%(ext)s" % r["item"])),
                        "https://www.youtube.com/watch?v=%s" % r["video_id"]])


def cut():
    CLIPS.mkdir(parents=True, exist_ok=True)
    meta = pd.read_excel(MCSD_DIR / "Files" / "MCSD_v1.xlsx")
    made = 0
    for _, r in meta.iterrows():
        out = CLIPS / ("%s.wav" % r["utterance_id"])
        src = EPISODES / ("%s.wav" % r["item"])
        if not out.exists() and src.exists():
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ss", str(r["start_time"]),
                            "-to", str(r["end_time"]), "-c", "copy", str(out)], capture_output=True)
        made += out.exists()
    print("%d of %d clips in %s" % (made, len(meta), CLIPS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "cut", "all"], nargs="?", default="all")
    a = ap.parse_args()
    if a.step in ("download", "all"):
        download()
    if a.step in ("cut", "all"):
        cut()
