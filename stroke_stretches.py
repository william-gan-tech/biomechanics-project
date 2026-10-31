"""
Phase 8d (10/31): candidate stretches for stroke rate, and sheets for the
researcher to confirm them.

From the whole-race pose (race_pose.py), a candidate stretch is a run of
frames that are live (race clock on screen, so no replays), within one
camera shot, with pose found and leg length >= MIN_LEG_PX (the Phase 7 size
rule), lasting at least MIN_FRAMES. Gaps of up to MAX_GAP frames without
pose are bridged.

Who the pose model followed, and whether it is a straightaway, is not known
automatically. Each stretch becomes one row of six frames on a sheet, with
the tracked hips (green) and feet (orange = left, blue = right) drawn, for
the researcher to label in stroke_rate/stretch_labels.csv:
    skater    the named skater, "other" (opponent, official), or "both"
    phase     straight, corner, or mixed (then use_from / use_to give the
              straightaway part)

Output: stroke_rate/stretch_sheets/<pair>_NN.jpg, stroke_rate/stretch_labels.csv

Usage:
    python -m stroke_stretches
"""

import os
import cv2
import numpy as np
import pandas as pd

from doublepush_measures import MIN_LEG_PX
from lap_graphics import VIDEO, WINDOWS

MIN_FRAMES = 60     # 2 s: room for 3+ set-downs at ~0.9 s per stroke
MAX_GAP = 5
PER_ROW = 6
ROWS_PER_SHEET = 8
OUT_DIR = "stroke_rate/stretch_sheets"
LABELS = "stroke_rate/stretch_labels.csv"
PAIR_SKATERS = {"pair01_kramer": "Kramer (orange NED) vs Thorup (red DEN)",
                "pair02_rijhnen": "Rijhnen (black GER) vs Lehman (white USA)",
                "pair05_roest": "Roest (orange NED) vs Giovannini (ITA)",
                "pair06_bergsma": "Bergsma (orange NED) vs Ichinohe (white JPN)",
                "pair09_bloemen": "Bloemen (black/red CAN) vs Zakharov (blue/red ROC)",
                "pair10_vanderpoel": "van der Poel (blue/yellow SWE) vs Swings (yellow/black BEL)"}


def stretches(d):
    ok = (d.leg_px >= MIN_LEG_PX) & d.live
    out = []
    for seq, g in d.groupby("cut_seq"):
        fr = g.frame[ok[g.index]].values
        if len(fr) == 0:
            continue
        start = prev = fr[0]
        for f in list(fr[1:]) + [None]:
            if f is None or f - prev > MAX_GAP + 1:
                if prev - start + 1 >= MIN_FRAMES:
                    out.append((int(start), int(prev)))
                if f is not None:
                    start = f
            if f is not None:
                prev = f
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, want = [], {}
    for pair in WINDOWS:
        d = pd.read_csv(f"stroke_rate/race_{pair}.csv")
        dd = d.set_index("frame")
        for i, (a, b) in enumerate(stretches(d), 1):
            sid = f"{pair[:6]}-{i:02d}"
            rows.append({"stretch": sid, "pair": pair, "start_frame": a, "end_frame": b,
                         "seconds": round((b - a + 1) / 30, 1), "skater": "", "phase": "",
                         "use_from": "", "use_to": "", "note": ""})
            for f in np.linspace(a, b, PER_ROW).astype(int):
                r = dd.loc[f] if f in dd.index else None
                want.setdefault(int(f), []).append((sid, r))
    lab = pd.DataFrame(rows)
    if not os.path.exists(LABELS):
        lab.to_csv(LABELS, index=False)
    thumbs = {}
    cap = cv2.VideoCapture(VIDEO)
    last, n = max(want), -1
    while n < last:
        if n + 1 not in want:
            cap.grab(); n += 1
            continue
        ok, f = cap.read(); n += 1
        if not ok:
            break
        for sid, r in want[n]:
            img = f.copy()
            if r is not None and not pd.isna(r.get("leg_px", np.nan)):
                hx, hy = int(r.hip_x), int(r.hip_y)
                cv2.circle(img, (hx, hy), 10, (0, 200, 0), -1)
                cv2.circle(img, (int(r.hip_x + r.lf), int(r.lf_y)), 9, (14, 127, 255), -1)
                cv2.circle(img, (int(r.hip_x + r.rf), int(r.rf_y)), 9, (180, 119, 31), -1)
            t = cv2.resize(img, (320, 180))
            cv2.putText(t, str(n), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            thumbs.setdefault(sid, []).append(t)
    cap.release()
    for pair, g in lab.groupby("pair", sort=False):
        ids = list(g.stretch)
        for k in range(0, len(ids), ROWS_PER_SHEET):
            band = []
            for sid in ids[k:k + ROWS_PER_SHEET]:
                r = g[g.stretch == sid].iloc[0]
                head = np.full((26, 320 * PER_ROW, 3), 255, np.uint8)
                cv2.putText(head, f"{sid}   frames {r.start_frame}-{r.end_frame}   {r.seconds} s",
                            (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                band += [head, np.hstack(thumbs[sid])]
            title = np.full((34, 320 * PER_ROW, 3), 230, np.uint8)
            cv2.putText(title, PAIR_SKATERS[pair], (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(OUT_DIR, f"{pair}_{k // ROWS_PER_SHEET:02d}.jpg"), np.vstack([title] + band))
    print(lab.groupby("pair").size().to_string(), "\n", len(lab), "stretches")


if __name__ == "__main__":
    main()
