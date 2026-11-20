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

11/15: the researcher's check of the first sheet found 7 of 43 sampled frames
with a box on the wrong person, nearly all in wide shots. The phase model
uses only frames inside the skater's marked race, with the race clock on
screen, leg at least MIN_LEG_PX (decided 11/15 to keep 150 px) and the same
track for at least MIN_RUN frames (usable()). --check-close draws a sheet of
those frames only, for the researcher to check identity where it matters.

The close-up check still found the lap-counter official (yellow jacket,
black trousers, close to Swings' colours) taken as Swings for a whole
stretch, so colour cannot settle identity alone. --runs puts one picture of
each stretch of usable frames (same track, steady) on a sheet, numbered, and
writes phase9/runs_<pair>.csv for the researcher to mark wrong ones
(identity column: ok / wrong). Frames in wrong runs are not used.

11/16: colour is now read only inside the torso the pose model finds
(shoulders to hips) on each candidate box, so the boards behind a skater no
longer count; the researcher's stretch check on pair 10 is its test.

Usage:
    python -m skater_pose --pair pair10_vanderpoel
    python -m skater_pose --pair pair10_vanderpoel --check-close
    python -m skater_pose --pair pair10_vanderpoel --runs
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

TRACKS = {"pair10_vanderpoel": "dual_tracking/beijing_p10_vanderpoel_swings_tracks.csv",
          "pair09_bloemen": "dual_tracking/beijing_p09_bloemen_zakharov_tracks.csv",
          "pair06_bergsma": "dual_tracking/beijing_p06_bergsma_ichinohe_tracks.csv"}
# torso colour in HSV (OpenCV hue 0-180); set per pair from the suits on screen
COLOURS = {"yellow": ((18, 110, 120), (35, 255, 255)), "blue": ((100, 90, 40), (125, 255, 255)),
           "black": ((0, 0, 0), (180, 255, 60)), "orange": ((0, 120, 100), (17, 255, 255)),
           "white": ((0, 0, 190), (180, 40, 255))}
SUITS = {"pair10_vanderpoel": {"Bart Swings": "yellow", "Nils van der Poel": "blue"},
         "pair09_bloemen": {"Ted-Jan Bloemen": "black", "Ruslan Zakharov": "blue"},
         "pair06_bergsma": {"Jorrit Bergsma": "orange", "Seitaro Ichinohe": "white"}}
MIN_SCORE = 0.08     # share of upper-half pixels in a suit colour, to be a candidate at all
MIN_TORSO = 0.30     # share of torso pixels (inside shoulders and hips) in the suit colour
MAX_CANDS = 4        # largest candidate boxes per frame that get pose
MIN_BOX_H = 60
PAD = 0.15
MIN_RUN = 15
GAP = 5              # 11/19: misses of up to GAP frames do not break a stretch if the same person returns
CHECK_EVERY = 300


def torso_scores(f, box, suits):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    t = f[int(max(0, y1)):int(y1 + 0.55 * h), int(max(0, x1 + 0.15 * w)):int(x1 + 0.85 * w)]
    if t.size == 0:
        return {k: 0.0 for k in suits}
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    return {k: float(cv2.inRange(hsv, *COLOURS[c]).mean() / 255) for k, c in suits.items()}


def torso_colour(crop, lm, suits):
    """Share of pixels in each suit colour inside the shoulder-hip quadrilateral of a pose."""
    h, w = crop.shape[:2]
    quad = np.array([[lm[i].x * w, lm[i].y * h] for i in (11, 12, 24, 23)], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)
    if mask.sum() < 255 * 30:
        return {k: 0.0 for k in suits}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return {k: float(cv2.inRange(hsv, *COLOURS[c])[mask > 0].mean() / 255) for k, c in suits.items()}


def short(name):
    return name.split()[-1].lower()


def usable(d, marks, gap=GAP):
    """Frames the phase model uses: inside the marked race, live, normal size, a steady track
    (MIN_RUN frames where the skater was found; misses of up to `gap` frames bridged, gap=0 as in round 1)."""
    from doublepush_measures import MIN_LEG_PX
    d = d.copy()
    tid = d.track_id.ffill(limit=gap) if gap else d.track_id
    d["run"] = (tid != tid.shift()).cumsum()
    real = d.track_id.notna()
    d["run_len"] = real.groupby(d.run).transform("sum")
    ok = d.frame.between(marks.frame.min(), marks.frame.max()) & d.live & (d.leg_px >= MIN_LEG_PX)
    return d[ok & real & (d.run_len >= MIN_RUN)]


def run_sheet(pair):
    marks = pd.read_csv("phase9/phase_marks.csv")
    marks = marks[(marks.pair == pair) & (marks.use == "yes")]
    rows = []
    for sk in SUITS[pair]:
        d = usable(pd.read_csv(f"phase9/pose_{pair}_{short(sk)}.csv"), marks[marks.skater == sk])
        for _, g in d.groupby("run"):
            mid = g.iloc[len(g) // 2]
            rows.append({"skater": sk, "first": int(g.frame.min()), "last": int(g.frame.max()), "frames": len(g),
                         "track_id": int(mid.track_id), "pic_frame": int(mid.frame),
                         "box": (mid.box_x1, mid.box_y1, mid.box_x2, mid.box_y2)})
    rows.sort(key=lambda r: r["first"])
    for i, r in enumerate(rows, 1):
        r["run"] = f"R{i:02d}"
    want = {}
    for r in rows:
        want.setdefault(r["pic_frame"], []).append(r)
    a = WINDOWS[pair][0]
    cap = cv2.VideoCapture(os.path.join(CLIPS, pair + ".mp4"))
    tiles = {}
    for n_ in range(a, max(want) + 1):
        ok, f = cap.read()
        if not ok:
            break
        for r in want.get(n_, []):
            t = f.copy()
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(t, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)
            cv2.rectangle(t, (0, 0), (1280, 64), (0, 0, 0), -1)
            cv2.putText(t, f"{r['run']}  {short(r['skater'])}  {r['frames']} frames", (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            tiles[r["run"]] = cv2.resize(t, (480, 270))
    cap.release()
    order = [tiles[r["run"]] for r in rows]
    while len(order) % 4:
        order.append(np.full_like(order[0], 255))
    for k in range(0, len(order), 32):
        part = order[k:k + 32]
        cv2.imwrite(f"phase9/runs_{pair}_{k // 32:02d}.jpg", np.vstack([np.hstack(part[i:i + 4]) for i in range(0, len(part), 4)]))
    t = pd.DataFrame([{k: v for k, v in r.items() if k != "box"} for r in rows])
    t["identity"] = ""
    t[["run", "skater", "first", "last", "frames", "track_id", "pic_frame", "identity"]].to_csv(f"phase9/runs_{pair}.csv", index=False)
    print(len(rows), "runs on", (len(order) + 31) // 32, "sheets")


def check_close(pair, n=48):
    marks = pd.read_csv("phase9/phase_marks.csv")
    marks = marks[(marks.pair == pair) & (marks.use == "yes")]
    picks = []
    for sk in SUITS[pair]:
        d = usable(pd.read_csv(f"phase9/pose_{pair}_{short(sk)}.csv"), marks[marks.skater == sk])
        take = d.iloc[np.linspace(0, len(d) - 1, min(len(d), n // len(SUITS[pair]))).astype(int)]
        picks += [(int(r.frame), sk, (r.box_x1, r.box_y1, r.box_x2, r.box_y2)) for r in take.itertuples()]
    picks.sort()
    want = {}
    for f, sk, bx in picks:
        want.setdefault(f, []).append((sk, bx))
    a = WINDOWS[pair][0]
    cap = cv2.VideoCapture(os.path.join(CLIPS, pair + ".mp4"))
    tiles = []
    for n_ in range(a, max(want) + 1):
        ok, f = cap.read()
        if not ok:
            break
        for sk, (x1, y1, x2, y2) in want.get(n_, []):
            t = f.copy()
            cv2.rectangle(t, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            cv2.putText(t, f"{n_}  {short(sk)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
            tiles.append(cv2.resize(t, (480, 270)))
    cap.release()
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    out = f"phase9/pose_check_close_{pair}.jpg"
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(out, len(picks), "boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(TRACKS))
    ap.add_argument("--check-close", action="store_true")
    ap.add_argument("--runs", action="store_true")
    args = ap.parse_args()
    if args.runs:
        run_sheet(args.pair)
        return
    if args.check_close:
        check_close(args.pair)
        return
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
        boxes = [bx for bx in by_frame.get(n, []) if bx[3] - bx[1] >= MIN_BOX_H]
        # candidates: some suit colour in the upper half; the largest few get pose, and colour is then
        # read inside the torso only
        cands = [bx for bx in boxes if max(torso_scores(f, bx[:4], suits).values()) >= MIN_SCORE]
        cands = sorted(cands, key=lambda bx: -(bx[2] - bx[0]) * (bx[3] - bx[1]))[:MAX_CANDS]
        posed = []
        for bx in cands:
            x1, y1, x2, y2, tid = bx
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
            p = pose_row(lm, cx2 - cx1, cy2 - cy1)
            p["hip_x"] += cx1
            for k in ("hip_y", "lf_y", "rf_y"):
                p[k] += cy1
            posed.append((torso_colour(crop, lm, suits), bx, p))
        taken, found = set(), {}
        for sk in sorted(suits, key=lambda k: -max([s[k] for s, _, _ in posed] or [0])):
            other = [k for k in suits if k != sk]
            ok_ = [((bx[2] - bx[0]) * (bx[3] - bx[1]), i) for i, (s, bx, _) in enumerate(posed)
                   if i not in taken and s[sk] >= MIN_TORSO and all(s[sk] > s[o] for o in other)]
            if ok_:
                i = max(ok_)[1]
                taken.add(i); found[sk] = (posed[i][0][sk], posed[i][1], posed[i][2])
        for sk in suits:
            row = {"frame": n, "cut_seq": seq, "diff": diff, "live": live}
            if sk in found:
                score, (x1, y1, x2, y2, tid), p = found[sk]
                row.update(track_id=int(tid), colour_score=round(score, 3), box_x1=x1, box_y1=y1, box_x2=x2, box_y2=y2)
                row.update(p)
            rows[sk].append(row)
        if (n - a) % CHECK_EVERY == 0 and found:
            t = f.copy()
            for sk, (score, (x1, y1, x2, y2, tid), _) in found.items():
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
