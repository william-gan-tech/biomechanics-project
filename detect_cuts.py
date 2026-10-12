"""
Phase 6c (10/11): camera-cut detection, so no track piece crosses a cut.

Found by the researcher's review of the Inzell pieces: piece P3 (track 375)
followed Jilek, then -- after a camera cut in which Eitrem appeared at about
the same place on screen -- Eitrem. The tracker kept the ID, and the pure-
pieces rules didn't fire (no jump in box position or size).

An earlier cut detector (9/28, whole-frame colour histograms) missed cuts
between similar-looking arena shots. This one compares frames PIXEL BY PIXEL
(downscaled greyscale): a cut changes the whole picture at once, while panning
and zooming change it gradually. A frame is a cut if its mean absolute
difference from the previous frame is both large and well above the typical
difference around it.

Output: dual_tracking/<name>_cuts.csv (frames where a new shot starts).

Usage:
    python -m detect_cuts --video data/inzell_2026_5000m.mp4 --start 0 --end 9300 --name inzell_full_race
    python -m detect_cuts --batch dual_tracking/test_clips.csv
"""

import argparse
import csv
import cv2
import numpy as np
import pandas as pd

ABS_THRESH = 25.0     # mean absolute grey-level difference (0-255) to count as a candidate cut
REL_FACTOR = 3.0      # ... and this many times the local median difference
LOCAL = 12            # frames either side for the local median


def frame_diffs(video, start, end):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev, frames, diffs = None, [], []
    n = start
    while n <= end:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(np.nan if prev is None else float(np.mean(np.abs(g - prev))))
        frames.append(n)
        prev = g
        n += 1
    cap.release()
    return pd.DataFrame({"frame": frames, "diff": diffs})


def cuts_from_diffs(d):
    s = d["diff"]
    local = s.rolling(2 * LOCAL + 1, center=True, min_periods=3).median()
    is_cut = (s >= ABS_THRESH) & (s >= REL_FACTOR * local)
    return d.loc[is_cut, "frame"].astype(int).tolist()


def run(video, start, end, name):
    d = frame_diffs(video, start, end)
    cuts = cuts_from_diffs(d)
    pd.DataFrame({"cut_frame": cuts}).to_csv(f"dual_tracking/{name}_cuts.csv", index=False)
    return len(cuts), len(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--name")
    p.add_argument("--batch")
    a = p.parse_args()
    jobs = []
    if a.batch:
        with open(a.batch, encoding="utf-8-sig") as f:
            jobs = [(r["video"], int(r["start"]), int(r["end"]), r["name"]) for r in csv.DictReader(f)]
    else:
        jobs = [(a.video, a.start, a.end, a.name)]
    for v, s, e, n in jobs:
        k, frames = run(v, s, e, n)
        print(f"{n:30s} {k:4d} cuts in {frames} frames", flush=True)


if __name__ == "__main__":
    main()
