"""
Phase 9b (11/20): add shoulder and knee points to a per-skater pose file.

The researcher's observations on telling straights from corners (11/20): on
straights the feet do not cross, crossovers come with much more lean, and on
straights a lean comes with the other leg pushing out wide while in corners
the legs stay close during the lean. Lean needs the shoulders, which
skater_pose.py did not save (and the Phase 6 detector comparison needs knee
angles). This reruns the pose model on the same box of every row that
already has one, so which person was chosen, and so the researcher's
stretch checks, are unchanged; it only adds columns (full-frame pixels).

Usage:
    python -m augment_pose phase9/pose_pair09_bloemen_bloemen.csv --pair pair09_bloemen
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from doublepush_measures import MODEL
from mark_phases import race
from skater_pose import PAD

POINTS = {"lsh": 11, "rsh": 12, "lhip": 23, "rhip": 24, "lknee": 25, "rknee": 26, "lank": 27, "rank": 28}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_file")
    ap.add_argument("--pair", required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.pose_file)
    want = d[d.track_id.notna()].set_index("frame")
    r = race(args.pair)
    lmk = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE))
    cap = cv2.VideoCapture(r["video"])
    n = r["base"] - 1
    extra = {}
    last = int(want.index.max())
    while n < last:
        if n + 1 not in want.index:
            cap.grab(); n += 1
            continue
        ok, f = cap.read(); n += 1
        if not ok:
            break
        H, W = f.shape[:2]
        x1, y1, x2, y2 = want.loc[n, ["box_x1", "box_y1", "box_x2", "box_y2"]]
        pw, ph = PAD * (x2 - x1), PAD * (y2 - y1)
        cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        cx2, cy2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
        crop = f[cy1:cy2, cx1:cx2]
        if not crop.size:
            continue
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not res.pose_landmarks:
            continue
        lm = res.pose_landmarks[0]
        cw, ch = cx2 - cx1, cy2 - cy1
        extra[n] = {f"{k}_{a}": (lm[i].x * cw + cx1) if a == "x" else (lm[i].y * ch + cy1)
                    for k, i in POINTS.items() for a in ("x", "y")}
    cap.release(); lmk.close()
    e = pd.DataFrame.from_dict(extra, orient="index")
    e.index.name = "frame"
    d = d.drop(columns=[c for c in e.columns if c in d.columns]).merge(e, left_on="frame", right_index=True, how="left")
    d.to_csv(args.pose_file, index=False)
    print(args.pose_file, len(extra), "rows with the extra points")


if __name__ == "__main__":
    main()
