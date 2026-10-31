"""
Phase 8d (10/30): pose on every frame of whole races, for stroke rate across
a race.

Phase 5 straightaway calls cover only a few short stretches per skater, so
pose is run on every frame of each pair's race window (lap_graphics.WINDOWS)
in one pass through the video. Per frame, the same values as the Phase 7
tracker (doublepush_measures.track), plus:
  - diff     mean absolute grey-level change from the previous frame; a cut
             (as in detect_cuts.py, but using only earlier frames) restarts
             the pose model and starts a new cut_seq
  - live     the race clock box is on screen (bottom right); replays and
             slow motion are shown without it

Who is in frame is not known here; stretches are confirmed by the
researcher on sheets (stroke_stretches.py).

Output: stroke_rate/race_<pair>.csv

Usage:
    python -m race_pose
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from detect_cuts import ABS_THRESH, REL_FACTOR, LOCAL
from doublepush_measures import MODEL
from lap_graphics import VIDEO, WINDOWS


def clock_live(f):
    # dark navy Olympic-rings box beside the clock; checked on Roest's race strips (10/30):
    # present on every frame from the start to the finish, absent before the start
    box = f[650:682, 1066:1150].astype(int)
    navy = (box[..., 0] > 35) & (box[..., 0] > box[..., 2] + 25) & (box[..., 1] < 40) & (box[..., 2] < 40)
    return bool(navy.mean() > 0.3)


def pose_row(lm, W, H):
    X = lambda i: lm[i].x * W
    Y = lambda i: lm[i].y * H
    hip_x, hip_y = (X(23) + X(24)) / 2, (Y(23) + Y(24)) / 2
    leg = (np.hypot(X(23) - X(27), Y(23) - Y(27)) + np.hypot(X(24) - X(28), Y(24) - Y(28))) / 2
    return {"leg_px": leg, "hip_x": hip_x, "hip_y": hip_y,
            "lf": (X(29) + X(31)) / 2 - hip_x, "rf": (X(30) + X(32)) / 2 - hip_x,
            "lf_y": (Y(29) + Y(31)) / 2, "rf_y": (Y(30) + Y(32)) / 2,
            "foot_vis": min(lm[i].visibility for i in (29, 30, 31, 32))}


def main():
    opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.VIDEO)
    todo = {k: v for k, v in WINDOWS.items() if not os.path.exists(f"stroke_rate/race_{k}.csv")}
    if not todo:
        return
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = cap.get(3), cap.get(4)
    end = max(b for a, b in todo.values())
    state = {}
    n = -1
    while n < end:
        nxt = n + 1
        k = next((k for k, (a, b) in todo.items() if a <= nxt <= b), None)
        if k is None:
            cap.grab(); n += 1
            continue
        ok, f = cap.read(); n += 1
        if not ok:
            break
        if k not in state:
            state[k] = {"lmk": vision.PoseLandmarker.create_from_options(opts), "t0": n, "seq": 0,
                        "prev": None, "diffs": [], "rows": []}
        s = state[k]
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.nan if s["prev"] is None else float(np.mean(np.abs(g - s["prev"])))
        s["prev"] = g
        recent = [x for x in s["diffs"][-LOCAL:] if not np.isnan(x)]
        if not np.isnan(diff) and diff >= ABS_THRESH and diff >= REL_FACTOR * (np.median(recent) if recent else 0):
            s["lmk"].close(); s["lmk"] = vision.PoseLandmarker.create_from_options(opts)
            s["t0"], s["seq"] = n, s["seq"] + 1
        s["diffs"].append(diff)
        row = {"frame": n, "cut_seq": s["seq"], "diff": diff, "live": clock_live(f)}
        res = s["lmk"].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)),
                                        int((n - s["t0"]) * 1000 / fps))
        if res.pose_landmarks:
            row.update(pose_row(res.pose_landmarks[0], W, H))
        s["rows"].append(row)
        if n == todo[k][1]:
            s["lmk"].close()
            pd.DataFrame(s["rows"]).to_csv(f"stroke_rate/race_{k}.csv", index=False)
            print(k, "done", len(s["rows"]), "frames", flush=True)
            del state[k]
    cap.release()


if __name__ == "__main__":
    main()
