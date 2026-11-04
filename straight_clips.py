"""
Phase 8d (11/3): clips for marking where each race stretch is really straight.

Hand checks on 11/1 showed that stretches labelled straight from six
thumbnails can include corner crossovers at either end. Every stretch that
has a stroke rate in stroke_rate/stretch_rates.csv becomes a slowed clip with
the frame number and skater burned in. The lap is left off and the clips are
in shuffled order, so marking is done without knowing early from late.

Output: stroke_rate/clips/straight_<stretch>.mp4, stroke_rate/straight_marks.csv

Usage:
    python -m straight_clips
    python -m mark_setdowns --straight
"""

import os
import cv2
import pandas as pd

from lap_graphics import VIDEO

MARKS = "stroke_rate/straight_marks.csv"
OUT_DIR = "stroke_rate/clips"
SLOW = 4
SEED = 11


def main():
    r = pd.read_csv("stroke_rate/stretch_rates.csv")
    r = r[r.rate_spm.notna()].sort_values("from")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = -1
    for _, s in r.iterrows():
        while n < s["from"] - 1:
            cap.grab(); n += 1
        out = None
        while n < s["to"]:
            ok, f = cap.read(); n += 1
            if not ok:
                break
            if out is None:
                h, w = f.shape[:2]
                out = cv2.VideoWriter(os.path.join(OUT_DIR, f"straight_{s.stretch}.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), fps / SLOW, (w, h))
            cv2.rectangle(f, (0, 0), (w, 44), (0, 0, 0), -1)
            cv2.putText(f, f"frame {n}   {s.skater}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out.write(f)
        if out is not None:
            out.release()
        print(s.stretch, s["to"] - s["from"] + 1, "frames")
    cap.release()
    if not os.path.exists(MARKS):
        t = pd.DataFrame({"clip": "straight_" + r.stretch, "skater": r.skater, "start_frame": r["from"],
                          "end_frame": r["to"], "straight_frames": "", "note": ""})
        t.sample(frac=1, random_state=SEED).to_csv(MARKS, index=False)


if __name__ == "__main__":
    main()
