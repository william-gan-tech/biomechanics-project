"""
Phase 9b (11/14): pose per named skater across a race, for the phase model.

The Phase 8 race pose (race_pose.py) follows whichever person the pose model
picks in each frame, often the skater nearer the camera. The phase model
needs the labelled skater's own pose, so here:
  1. every person box comes from the Phase 6 tracks (YOLO + ByteTrack,
     dual_tracking/<name>_tracks.csv; checked 11/14 to line up with the
     pair's clip frame for frame),
  2. each box is given to a skater by the colour of its upper half (SUITS);
     of the boxes with enough of that colour, the largest is taken, since
     officials at the boards are usually smaller in frame than the skaters
     (first version took the best colour score above 0.25 and missed Swings,
     who skates bent forward, and picked up officials in blue and yellow),
  3. pose is run on a padded crop of that box and stored in full-frame
     coordinates, with the same values as race_pose.py.
Frames where a skater's colour is not found are kept with no pose.

Output: phase9/pose_<pair>_<skater>.csv, and phase9/pose_check_<pair>.jpg
(sampled frames with each skater's box drawn, for the researcher to confirm
identity).

Usage:
    python -m skater_pose --pair pair10_vanderpoel
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from detect_cuts import ABS_THRESH, REL_FACTOR, LOCAL
from doublepush_measures import MODEL
from lap_graphics import WINDOWS
from mark_phases import CLIPS
from race_pose import clock_live, pose_row

TRACKS = {"pair10_vanderpoel": "dual_tracking/beijing_p10_vanderpoel_swings_tracks.csv"}
# torso colour in HSV (OpenCV hue 0-180); set per pair from the suits on screen
COLOURS = {"yellow": ((18, 110, 120), (35, 255, 255)), "blue": ((100, 90, 40), (125, 255, 255))}
SUITS = {"pair10_vanderpoel": {"Bart Swings": "yellow", "Nils van der Poel": "blue"}}
MIN_SCORE = 0.08     # share of upper-half pixels in the suit colour
MIN_BOX_H = 60
PAD = 0.15
CHECK_EVERY = 300


def torso_scores(f, box, suits):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    t = f[int(max(0, y1)):int(y1 + 0.55 * h), int(max(0, x1 + 0.15 * w)):int(x1 + 0.85 * w)]
    if t.size == 0:
        return {k: 0.0 for k in suits}
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    return {k: float(cv2.inRange(hsv, *COLOURS[c]).mean() / 255) for k, c in suits.items()}


def short(name):
    return name.split()[-1].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(TRACKS))
    args = ap.parse_args()
    pair, suits = args.pair, SUITS[args.pair]
    a, b = WINDOWS[pair]
    tracks = pd.read_csv(TRACKS[pair])
    by_frame = {f: g[["x1", "y1", "x2", "y2", "track_id"]].values for f, g in tracks.groupby("frame")}
    opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.IMAGE)
    lmk = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(os.path.join(CLIPS, pair + ".mp4"))
    rows = {k: [] for k in suits}
    prev, diffs, seq, tiles = None, [], 0, []
    for n in range(a, b + 1):
        ok, f = cap.read()
        if not ok:
            break
        H, W = f.shape[:2]
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.nan if prev is None else float(np.mean(np.abs(g - prev)))
        prev = g
        recent = [x for x in diffs[-LOCAL:] if not np.isnan(x)]
        if not np.isnan(diff) and diff >= ABS_THRESH and diff >= REL_FACTOR * (np.median(recent) if recent else 0):
            seq += 1
        diffs.append(diff)
        live = clock_live(f)
        boxes = by_frame.get(n, [])
        scored = [(torso_scores(f, bx[:4], suits), bx) for bx in boxes]
        taken = set()
        found = {}
        for sk in sorted(suits, key=lambda k: -max([s[k] for s, _ in scored] or [0])):
            other = [k for k in suits if k != sk]
            cands = [((bx[2] - bx[0]) * (bx[3] - bx[1]), i) for i, (s, bx) in enumerate(scored)
                     if i not in taken and s[sk] >= MIN_SCORE and bx[3] - bx[1] >= MIN_BOX_H
                     and all(s[sk] > s[o] for o in other)]
            if cands:
                i = max(cands)[1]
                taken.add(i); found[sk] = (scored[i][0][sk], scored[i][1])
        for sk in suits:
            row = {"frame": n, "cut_seq": seq, "diff": diff, "live": live}
            if sk in found:
                score, (x1, y1, x2, y2, tid) = found[sk]
                row.update(track_id=int(tid), colour_score=round(score, 3), box_x1=x1, box_y1=y1, box_x2=x2, box_y2=y2)
                pw, ph = PAD * (x2 - x1), PAD * (y2 - y1)
                cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
                cx2, cy2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
                crop = f[cy1:cy2, cx1:cx2]
                if crop.size:
                    res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    if res.pose_landmarks:
                        p = pose_row(res.pose_landmarks[0], cx2 - cx1, cy2 - cy1)
                        for k in ("hip_x", "lf_y", "rf_y"):
                            p[k] += cx1 if k == "hip_x" else cy1
                        p["hip_y"] += cy1
                        row.update(p)
            rows[sk].append(row)
        if (n - a) % CHECK_EVERY == 0 and found:
            t = f.copy()
            for sk, (score, (x1, y1, x2, y2, tid)) in found.items():
                cv2.rectangle(t, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(t, f"{short(sk)} {score:.2f}", (int(x1), max(30, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(t, str(n), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
            tiles.append(cv2.resize(t, (480, 270)))
        if (n - a) % 2000 == 0:
            print(n, flush=True)
    cap.release(); lmk.close()
    os.makedirs("phase9", exist_ok=True)
    for sk, r in rows.items():
        d = pd.DataFrame(r)
        d.to_csv(f"phase9/pose_{pair}_{short(sk)}.csv", index=False)
        print(sk, len(d), "frames, identified", round(d.track_id.notna().mean(), 3) if "track_id" in d else 0,
              "pose", round(d.leg_px.notna().mean(), 3) if "leg_px" in d else 0)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(f"phase9/pose_check_{pair}.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main()
