"""
Phase 9a (11/13): a sheet of frames around a skater's phase marks, each with
the label the marks give it, so the researcher can confirm them by eye.

Usage:
    python -m phase_check_sheet --pair pair10_vanderpoel --skater "Nils van der Poel" --from 180400 --to 180700 --step 15
Output: phase9/check_<pair>_<from>.jpg
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd

from lap_graphics import WINDOWS
from mark_phases import MARKS, CLIPS

COLOURS = {"straight": (60, 160, 60), "corner": (40, 40, 220), "straight in corner": (0, 160, 230), "before start": (120, 120, 120)}


def label_at(marks, f):
    state = "before start"
    for _, m in marks.iterrows():
        if m.frame > f:
            break
        state = {"start": "straight", "entry": "corner", "exit": "straight", "in_straight": "straight in corner",
                 "in_cross": "corner", "finish": "after finish"}[m.event]
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--skater", required=True)
    ap.add_argument("--from", dest="a", type=int, required=True)
    ap.add_argument("--to", dest="b", type=int, required=True)
    ap.add_argument("--step", type=int, default=15)
    args = ap.parse_args()
    d = pd.read_csv(MARKS)
    marks = d[(d.pair == args.pair) & (d.skater == args.skater)].sort_values("frame")
    want = sorted(set(range(args.a, args.b + 1, args.step)) | set(marks.frame[(marks.frame >= args.a) & (marks.frame <= args.b)]))
    base = WINDOWS[args.pair][0]
    cap = cv2.VideoCapture(os.path.join(CLIPS, args.pair + ".mp4"))
    n, tiles = base - 1, []
    while n < want[-1]:
        ok, f = cap.read(); n += 1
        if not ok:
            break
        if n not in want:
            continue
        t = cv2.resize(f, (480, 270))
        lab = label_at(marks, n)
        mark = marks[marks.frame == n]
        cv2.rectangle(t, (0, 0), (480, 30), COLOURS.get(lab, (0, 0, 0)), -1)
        text = f"{n}  {lab}" + (f"   MARK: {mark.event.iloc[0]}" if len(mark) else "")
        cv2.putText(t, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    cap.release()
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    out = f"phase9/check_{args.pair}_{args.a}.jpg"
    cv2.imwrite(out, sheet)
    print(out, len(want), "frames")


if __name__ == "__main__":
    main()
