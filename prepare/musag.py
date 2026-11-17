import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yt_dlp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MUSAG_DIR

VIDEOS = MUSAG_DIR / "videos"
SEGMENTS = MUSAG_DIR / "segments"


def seconds(t):
    if "." in t:
        return sum(float(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))
    t = datetime.strptime(t, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def main():
    df = pd.read_csv(MUSAG_DIR / "dataset.csv")
    VIDEOS.mkdir(parents=True, exist_ok=True)
    SEGMENTS.mkdir(parents=True, exist_ok=True)
    for url in df["url"].unique():
        vid = url.split("v=")[-1]
        if (VIDEOS / ("%s.mp4" % vid)).exists():
            continue
        opts = {"format": "bestvideo+bestaudio/best", "merge_output_format": "mp4",
                "outtmpl": str(VIDEOS / ("%s.%%(ext)s" % vid))}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print("%s: %s" % (vid, e))
    made = 0
    for _, r in df.iterrows():
        out = SEGMENTS / ("%s.wav" % r["id"])
        src = VIDEOS / ("%s.mp4" % r["url"].split("v=")[-1])
        if not out.exists() and src.exists():
            start = seconds(r["section_timestamp_start"])
            end = seconds(r["section_timestamp_end"])
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-t", str(end - start),
                            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out)],
                           capture_output=True)
        made += out.exists()
    print("%d of %d segments in %s" % (made, len(df), SEGMENTS))


if __name__ == "__main__":
    main()
