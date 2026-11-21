"""
Phase 9 (11/20): how close does a broadcast film the skaters on the straights?

Rounds 1 and 2 on the Beijing 2022 broadcast had close-ups almost only in
corners (straights 90 and 16 frames). Before marking another race, each
downloaded race gets a sheet of evenly spaced frames with the leg length of
the largest person found (pose model on the whole frame), so the researcher
can judge which broadcast shows straights up close.

Output: phase9/scout_<video>.jpg, phase9/scout_summary.csv

Usage:
    python -m broadcast_scout
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from doublepush_measures import MODEL, MIN_LEG_PX
from race_pose import pose_row

VIDEOS = ["data/inzell_2026_5000m.mp4", "data/ghiotto_10000m_calgary2025.mp4",
          "data/vanderpoel_10000m_stavanger.mp4", "data/semirunny_5000m_tomaszow2026.mp4"]
N = 48


def main():
    lmk = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE, num_poses=3))
    rows = []
    for v in VIDEOS:
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set(np.linspace(total * 0.05, total * 0.95, N).astype(int))
        tiles, n = [], -1
        while n < max(want):
            if n + 1 not in want:
                cap.grab(); n += 1
                continue
            ok, f = cap.read(); n += 1
            if not ok:
                break
            H, W = f.shape[:2]
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            legs = [pose_row(lm, W, H)["leg_px"] * 720 / H for lm in res.pose_landmarks]   # in 720p pixels
            leg = max(legs) if legs else 0
            rows.append({"video": os.path.basename(v), "frame": n, "leg_px_720p": round(leg)})
            t = cv2.resize(f, (480, 270))
            col = (0, 160, 0) if leg >= MIN_LEG_PX else (0, 0, 200)
            cv2.rectangle(t, (0, 0), (480, 28), col, -1)
            cv2.putText(t, f"{n}   leg {leg:.0f}px", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(t)
        cap.release()
        while len(tiles) % 6:
            tiles.append(np.full_like(tiles[0], 255))
        name = os.path.splitext(os.path.basename(v))[0]
        cv2.imwrite(f"phase9/scout_{name}.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
        print(name, "done", flush=True)
    lmk.close()
    d = pd.DataFrame(rows)
    d.to_csv("phase9/scout_summary.csv", index=False)
    print(d.groupby("video").leg_px_720p.agg(sampled="size", median="median",
                                              close_up_share=lambda x: round((x >= MIN_LEG_PX).mean(), 2)))


if __name__ == "__main__":
    main()
