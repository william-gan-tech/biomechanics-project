"""
Phase 8b: clips for counting strokes by eye.

Takes the Beijing 2022 5000m straightaways already called by the researcher
in Phase 5 (labeled_segments.csv), keeps those at least MIN_SEC long, and
splits them by skater, fixed before any automatic counting:
  - tune: Kramer, Roest (the counter may be adjusted on these)
  - test: Bergsma, Bloemen, Rijhnen (never used to adjust the counter)

Each segment becomes a slowed clip (SLOW x slower) with the video frame
number burned in, so the frame of each skate set-down can be written down.
Also writes stroke_rate/hand_counts.csv for those frames to be filled in.

Reads frames sequentially (seeking is unreliable in yt-dlp downloads).

Usage:
    python -m stroke_clips
"""

import os
import cv2
import pandas as pd

VIDEO = "data/beijing_2022_5000m.mp4"
OUT_DIR = "stroke_rate/clips"
COUNTS = "stroke_rate/hand_counts.csv"
MIN_SEC = 2.4
SLOW = 4
TUNE = {"Sven Kramer", "Patrick Roest"}


def segments():
    d = pd.read_csv("labeled_segments.csv")
    d = d[d.video_path.str.contains("beijing") & (d.phase == "straightaway")].copy()
    d["lap"] = d.notes.str.extract(r"lap (\d+)", expand=False).astype(int)
    d = d[(d.end_frame - d.start_frame) / 30 >= MIN_SEC]
    d["set"] = d.skater.map(lambda s: "tune" if s in TUNE else "test")
    d = d.sort_values("start_frame").reset_index(drop=True)
    d["clip"] = [f"{s}_{i:02d}" for i, s in enumerate(d.set)]
    return d


def main():
    d = segments()
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = -1
    for _, s in d.iterrows():
        while n < s.start_frame - 1:
            cap.grab(); n += 1
        out = None
        while n < s.end_frame:
            ok, f = cap.read(); n += 1
            if not ok:
                break
            if out is None:
                h, w = f.shape[:2]
                out = cv2.VideoWriter(os.path.join(OUT_DIR, s["clip"] + ".mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), fps / SLOW, (w, h))
            label = f"frame {n}   {s.skater}, lap {s.lap}"
            cv2.rectangle(f, (0, 0), (w, 44), (0, 0, 0), -1)
            cv2.putText(f, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out.write(f)
        if out is not None:
            out.release()
        print(s["clip"], s.skater, s.lap, s.start_frame, s.end_frame)
    cap.release()
    if not os.path.exists(COUNTS):
        t = d[["clip", "set", "skater", "lap", "start_frame", "end_frame"]].copy()
        t["setdown_frames"] = ""
        t["note"] = ""
        t.to_csv(COUNTS, index=False)


if __name__ == "__main__":
    main()
