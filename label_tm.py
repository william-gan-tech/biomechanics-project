"""
Phase 9 (11/21): pick and label skater pictures for the Teachable Machine
corner/straight model.

The researcher wanted to choose many training pictures by eye. --pool cuts
every STEP-th frame where a Beijing training skater was found inside the
marked race (leg at least MIN_LEG, smaller than the model's size rule, since
the researcher judges quality) into phase9/tm_pool/, with the label the marks
suggest. The player then shows them one at a time, large:
    C   corner          S   straight
    X   skip (unclear, wrong person, bad picture)
    U   undo the last   Q   save and quit (resumes where it stopped)
Pictures labelled C or S are copied to phase9/tm_labelled/<label>/ for
Teachable Machine. Calgary (the round 3 test race) is never in the pool.

Usage:
    python -m label_tm --pool      # once
    python -m label_tm             # label
"""

import os
import shutil
import argparse
import cv2
import numpy as np
import pandas as pd

from label_audit import frame_labels
from mark_phases import race
from skater_pose import PAD, short

POOL = "phase9/tm_pool"
OUT = "phase9/tm_labelled"
LOG = "phase9/tm_labels.csv"
STEP = 6
MIN_LEG = 80
SKATERS = [("pair10_vanderpoel", "Bart Swings", "phase9/pose_pair10_vanderpoel_swings_v1.csv"),
           ("pair10_vanderpoel", "Nils van der Poel", "phase9/pose_pair10_vanderpoel_poel_v1.csv"),
           ("pair09_bloemen", "Ted-Jan Bloemen", "phase9/pose_pair09_bloemen_bloemen.csv"),
           ("pair06_bergsma", "Jorrit Bergsma", "phase9/pose_pair06_bergsma_bergsma.csv")]


def make_pool():
    os.makedirs(POOL, exist_ok=True)
    marks = pd.read_csv("phase9/phase_marks.csv")
    marks = marks[marks.use == "yes"]
    rows = []
    for pair, sk, pose in SKATERS:
        m = marks[(marks.pair == pair) & (marks.skater == sk)]
        lab = frame_labels(m)
        d = pd.read_csv(pose)
        d = d[d.track_id.notna() & (d.leg_px >= MIN_LEG) & d.frame.isin(lab.keys())]
        d = d.iloc[::STEP].set_index("frame")
        r = race(pair)
        cap = cv2.VideoCapture(r["video"])
        n = r["base"] - 1
        for target in d.index:
            while n < target - 1:
                cap.grab(); n += 1
            ok, f = cap.read(); n += 1
            if not ok:
                break
            x = d.loc[target]
            H, W = f.shape[:2]
            pw, ph = PAD * (x.box_x2 - x.box_x1), PAD * (x.box_y2 - x.box_y1)
            crop = f[int(max(0, x.box_y1 - ph)):int(min(H, x.box_y2 + ph)), int(max(0, x.box_x1 - pw)):int(min(W, x.box_x2 + pw))]
            name = f"{pair}_{short(sk)}_{target}.jpg"
            cv2.imwrite(os.path.join(POOL, name), crop)
            rows.append({"image": name, "skater": sk, "frame": target, "suggested": lab[target], "leg_px": round(x.leg_px)})
        cap.release()
        print(sk, sum(r["skater"] == sk for r in rows), flush=True)
    p = pd.DataFrame(rows).sample(frac=1, random_state=21)    # mixed order, so one race does not come in a block
    p.to_csv(os.path.join(POOL, "pool.csv"), index=False)
    print(len(p), "pictures;", p.suggested.value_counts().to_dict())


def show(img, text, sub, done, total):
    h, w = img.shape[:2]
    s = min(700 / h, 900 / w)
    big = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((800, 960, 3), 30, np.uint8)
    y0, x0 = 70 + (700 - big.shape[0]) // 2, (960 - big.shape[1]) // 2
    canvas[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
    cv2.putText(canvas, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, sub, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, f"{done} labelled of {total}   [C corner  S straight  X skip  U undo  Q quit]", (12, 790),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.imshow("label for Teachable Machine", canvas)


def label():
    pool = pd.read_csv(os.path.join(POOL, "pool.csv"))
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=["image", "label"])
    done = list(log.itertuples(index=False, name=None))
    seen = {i for i, _ in done}
    todo = [r for r in pool.itertuples() if r.image not in seen]
    i = 0
    while i < len(todo):
        r = todo[i]
        img = cv2.imread(os.path.join(POOL, r.image))
        n_lab = sum(l in ("corner", "straight") for _, l in done)
        show(img, f"{r.skater}", f"marks say: {r.suggested}   (leg {r.leg_px} px)", n_lab, len(pool))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("c"), ord("C"), ord("s"), ord("S"), ord("x"), ord("X")):
            lab = {"c": "corner", "s": "straight", "x": "skip"}[chr(k).lower()]
            done.append((r.image, lab))
            if lab != "skip":
                os.makedirs(os.path.join(OUT, lab), exist_ok=True)
                shutil.copy(os.path.join(POOL, r.image), os.path.join(OUT, lab, r.image))
            i += 1
        elif k in (ord("u"), ord("U")) and done and i > 0:
            img_, lab = done.pop()
            if lab != "skip":
                os.remove(os.path.join(OUT, lab, img_))
            i -= 1
        elif k in (ord("q"), ord("Q")) or cv2.getWindowProperty("label for Teachable Machine", cv2.WND_PROP_VISIBLE) < 1:
            break
        pd.DataFrame(done, columns=["image", "label"]).to_csv(LOG, index=False)
    pd.DataFrame(done, columns=["image", "label"]).to_csv(LOG, index=False)
    cv2.destroyAllWindows()
    c = pd.Series([l for _, l in done]).value_counts().to_dict()
    print("labelled so far:", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true")
    args = ap.parse_args()
    make_pool() if args.pool else label()


if __name__ == "__main__":
    main()
