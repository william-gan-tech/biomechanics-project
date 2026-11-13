"""
Phase 9d (11/12): sheets of the broadcast's finish graphic.

lap_graphics.py finds the split graphic by its yellow rank box, which moves to
the far left once a skater finishes, so the finish times were mostly missed.
This scans from each pair's last split graphic to past the end of its window
for a yellow box at the left edge, and puts one strip per run on a sheet.

Output: stroke_rate/lap_sheets/<pair>_finish.jpg

Usage:
    python -m finish_graphics
"""

import os
import cv2
import numpy as np

from lap_graphics import VIDEO, OUT_DIR, STEP

# from the last split graphic on each Phase 8 sheet to 3000 frames (100 s) past the window
SCAN = {"pair01_kramer": (18550, 23500), "pair02_rijhnen": (33070, 37500), "pair05_roest": (76890, 80400),
        "pair06_bergsma": (129620, 134500), "pair09_bloemen": (175045, 180000), "pair10_vanderpoel": (190515, 195500)}


def has_finish(strip):
    roi = strip[:, 55:95].astype(int)
    return ((roi[..., 2] > 180) & (roi[..., 1] > 140) & (roi[..., 0] < 90)).sum() > 15


def main():
    hits = {k: [] for k in SCAN}
    cap = cv2.VideoCapture(VIDEO)
    end = max(b for a, b in SCAN.values())
    n = -1
    while n < end:
        nxt = n + 1
        live = [k for k, (a, b) in SCAN.items() if a <= nxt <= b and (nxt - a) % STEP == 0]
        if not live:
            cap.grab(); n += 1
            continue
        ok, f = cap.read(); n += 1
        if not ok:
            break
        strip = cv2.resize(f[560:720], (640, 80))
        if has_finish(strip):
            for k in live:
                hits[k].append((n, strip))
    cap.release()
    for k, h in hits.items():
        if not h:
            print(k, "no finish graphic found"); continue
        runs, cur = [], [h[0]]
        for x in h[1:]:
            if x[0] - cur[-1][0] > 3 * STEP:
                runs.append(cur); cur = [x]
            else:
                cur.append(x)
        runs.append(cur)
        tiles = []
        for r in runs:
            t = cv2.resize(r[len(r) // 2][1], (960, 120))
            cv2.putText(t, f"{r[0][0]}-{r[-1][0]}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            tiles.append(t)
        cv2.imwrite(os.path.join(OUT_DIR, k + "_finish.jpg"), np.vstack(tiles))
        print(k, len(runs), "runs")


if __name__ == "__main__":
    main()
