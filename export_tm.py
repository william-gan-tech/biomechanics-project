"""
Phase 9 (11/20): skater images for a Google Teachable Machine corner/straight
model (the researcher's idea; Tier 2 comparison with the pose model).

From the Beijing training skaters only (Calgary stays the round 3 test), every
STEP-th frame the phase model would use (marked race, normal size, identity
confirmed by the researcher's stretch check) is cut out around the skater's
box and saved by its label from the marks. The researcher then picks
good-quality examples of each for training; the test images are not picked.

Output: phase9/tm_images/<label>/<pair>_<skater>_<frame>.jpg

Usage:
    python -m export_tm
"""

import os
import cv2
import pandas as pd

from mark_phases import race
from phase_model import DATA, load
from skater_pose import PAD, short

OUT = "phase9/tm_images"
STEP = 5
SKATERS = [("pair10_vanderpoel", "Bart Swings"), ("pair10_vanderpoel", "Nils van der Poel"),
           ("pair09_bloemen", "Ted-Jan Bloemen"), ("pair06_bergsma", "Jorrit Bergsma")]


def main():
    for lab in ("corner", "straight"):
        os.makedirs(os.path.join(OUT, lab), exist_ok=True)
    counts = {}
    for pair, sk in SKATERS:
        d, _ = load(pair, sk, gap=5)
        d = d.sort_values("frame")
        d = d[d.groupby("label").cumcount() % STEP == 0].set_index("frame")
        r = race(pair)
        cap = cv2.VideoCapture(r["video"])
        n = r["base"] - 1
        last = int(d.index.max())
        while n < last:
            if n + 1 not in d.index:
                cap.grab(); n += 1
                continue
            ok, f = cap.read(); n += 1
            if not ok:
                break
            x = d.loc[n]
            H, W = f.shape[:2]
            pw, ph = PAD * (x.box_x2 - x.box_x1), PAD * (x.box_y2 - x.box_y1)
            crop = f[int(max(0, x.box_y1 - ph)):int(min(H, x.box_y2 + ph)), int(max(0, x.box_x1 - pw)):int(min(W, x.box_x2 + pw))]
            cv2.imwrite(os.path.join(OUT, x.label, f"{pair}_{short(sk)}_{n}.jpg"), crop)
            counts[(sk, x.label)] = counts.get((sk, x.label), 0) + 1
        cap.release()
    for k, v in sorted(counts.items()):
        print(k, v)


if __name__ == "__main__":
    main()
