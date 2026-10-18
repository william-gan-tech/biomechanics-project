"""
Phase 6f (10/17): 2D and 3D pose for the Phase 5 fatigue (5f) segments.

Phase 5 stored only 2D features. This runs the same pose model
(pose_landmarker_lite, whole frame, first person found, as in Phase 5) on the
frames of every early/late 5f segment and saves, per frame:
  - 2D trunk lean, right and left side (image plane, as in Phase 5)
  - 3D trunk lean, right and left side (angle of hip-to-shoulder line from
    vertical, MediaPipe world landmarks)
  - 2D and 3D knee angles

Frame numbers: Phase 5 segment frames are 1-based, so segment frame f is
read at video position f - 1. Each segment gets a fresh landmarker.

Output: phase6f_5f_segments_3d.csv

Usage:
    python -m extract_3d_segments
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from validate_fatigue_form_degradation import lap_segments
from piece_pose import angle3d
from utils.biomechanics_utils import calculate_angle

MODEL = "pose_landmarker_lite.task"
OUTPUT = "phase6f_5f_segments_3d.csv"


def lean2d(hip, sh):
    return abs(float(np.degrees(np.arctan2(sh[0] - hip[0], -(sh[1] - hip[1])))))


def lean3d(hip, sh):
    v = sh - hip
    return float(np.degrees(np.arccos(np.clip(-v[1] / (np.linalg.norm(v) + 1e-9), -1.0, 1.0))))


def frame_values(lm, wl, w, h):
    px = lambda i: np.array([lm[i].x * w, lm[i].y * h])
    P = lambda i: np.array([wl[i].x, wl[i].y, wl[i].z])
    return {
        "lean2d_r": lean2d(px(24), px(12)), "lean2d_l": lean2d(px(23), px(11)),
        "lean3d_r": lean3d(P(24), P(12)), "lean3d_l": lean3d(P(23), P(11)),
        "knee2d_r": calculate_angle(px(24).tolist(), px(26).tolist(), px(28).tolist()),
        "knee2d_l": calculate_angle(px(23).tolist(), px(25).tolist(), px(27).tolist()),
        "knee3d_r": angle3d(P(24), P(26), P(28)), "knee3d_l": angle3d(P(23), P(25), P(27)),
        "torso_px": float(np.linalg.norm((px(11) + px(12)) / 2 - (px(23) + px(24)) / 2)),
    }


def main():
    segs = lap_segments()
    print(f"{len(segs)} early/late 5f segments")
    opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.VIDEO)
    rows, caps = [], {}
    for i, r in enumerate(segs.itertuples(), 1):
        cap = caps.get(r.video_path) or caps.setdefault(r.video_path, cv2.VideoCapture(r.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        a, b = int(r.start_frame), int(r.end_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a - 1)
        n = 0
        with vision.PoseLandmarker.create_from_options(opts) as lmk:
            for f in range(a, b + 1):
                ok, img = cap.read()
                if not ok:
                    break
                res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                    data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                                           int((f - a) * 1000 / fps))
                if not res.pose_landmarks or not res.pose_world_landmarks:
                    continue
                v = frame_values(res.pose_landmarks[0], res.pose_world_landmarks[0], img.shape[1], img.shape[0])
                if v["torso_px"] < 10.0:          # same anomaly filter as Phase 5
                    continue
                rows.append({"skater": r.skater, "video_path": r.video_path, "phase": r.phase,
                             "stage": r.stage, "lap": r.lap, "start_frame": a, "end_frame": b, "frame": f, **v})
                n += 1
        print(f"[{i}/{len(segs)}] {r.skater} {r.phase} lap {r.lap} {a}-{b}: {n} frames", flush=True)
    for c in caps.values():
        c.release()
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
