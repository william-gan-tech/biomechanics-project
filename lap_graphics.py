"""
Phase 8a (10/30): sheets of the broadcast's lap-time graphic, for reading
each skater's 400 m splits.

The Beijing 2022 broadcast shows a split graphic every 400 m (skater,
cumulative time, lap time, difference to the leader) along the bottom of
the frame. Official lap times could not be found online (10/28), so they are
read from these graphics instead.

For each pair's race window, the bottom strip of every STEP-th frame is
checked for the graphic (the yellow rank box); consecutive frames with it
are grouped, and one strip per group goes on a sheet with its frame range.
One pass through the video covers every window (seeking is unreliable).

Output: stroke_rate/lap_sheets/<name>.jpg

Usage:
    python -m lap_graphics
"""

import os
import cv2
import numpy as np

VIDEO = "data/beijing_2022_5000m.mp4"
OUT_DIR = "stroke_rate/lap_sheets"
STEP = 15
# race windows (frames), from race_scan_beijing_2022_5000m/pairs_map.md
WINDOWS = {"pair01_kramer": (7000, 20500), "pair02_rijhnen": (22000, 34500),
           "pair05_roest": (66000, 77400), "pair06_bergsma": (119000, 131500),
           "pair09_bloemen": (163000, 177000), "pair10_vanderpoel": (180000, 192500)}


def has_graphic(strip):
    roi = strip[:, 270:300].astype(int)
    return ((roi[..., 2] > 180) & (roi[..., 1] > 140) & (roi[..., 0] < 90)).sum() > 15


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wins = sorted(WINDOWS.items(), key=lambda kv: kv[1][0])
    hits = {k: [] for k in WINDOWS}
    cap = cv2.VideoCapture(VIDEO)
    end = max(b for a, b in WINDOWS.values())
    n = -1
    while n < end:
        nxt = n + 1
        live = [k for k, (a, b) in wins if a <= nxt <= b and (nxt - a) % STEP == 0]
        if not live:
            cap.grab(); n += 1
            continue
        ok, f = cap.read(); n += 1
        if not ok:
            break
        strip = cv2.resize(f[560:720], (640, 80))
        if has_graphic(strip):
            for k in live:
                hits[k].append((n, strip))
    cap.release()
    for k, h in hits.items():
        if not h:
            print(k, "no graphics found"); continue
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
        cv2.imwrite(os.path.join(OUT_DIR, k + ".jpg"), np.vstack(tiles))
        print(k, len(runs), "graphics")


if __name__ == "__main__":
    main()
