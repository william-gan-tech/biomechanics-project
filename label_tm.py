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

11/22: round 3 found most straight errors in the second half of straights
(3+ s after the corner exit), and the researcher's straight pictures had few
of those. --late shows only pool pictures that the marks call straight and
that are at least LATE_S seconds after a corner exit (Beijing races only).

11/25: --upper builds and labels a separate set for an upper-body model (the
researcher's idea): frames where the skater was found but the feet are barely
visible (foot visibility below 0.3, the frames now judged "can't judge"), from
the four Beijing skaters and Calgary's Ghiotto (Inzell, a test race, is never
used), in their own pool, label list and folders.

Usage:
    python -m label_tm --pool      # once
    python -m label_tm             # label
    python -m label_tm --late      # only late-straight pictures
    python -m label_tm --upper --pool   # once: the upper-body pool
    python -m label_tm --upper          # label upper-body pictures
    python -m label_tm --more --pool    # once: more full-body pictures, all six marked skaters
    python -m label_tm --more           # label them (same keys and folders as the first full-body set)
    python -m label_tm --more --first calgary,inzell     # those groups' pictures first (11/27)
    python -m label_tm --upper --add inzell2026_eitrem   # add a race to the upper-body pool (11/27)
    python -m label_tm --upper --first inzell
    python -m label_tm --identity --pool   # once: people boxes for skater identity (9g)
    python -m label_tm --identity          # R racer, N not a racer, X skip

11/27, --more: to improve the full-body model as a whole, a second pool from
all six marked skaters on all three broadcasts (confirmed identity, normal
size), half of it within 2 s of a corner entry or exit, where the models find
it hardest, and half anywhere else. Labels go to the same list and folders as
the first full-body set, so both are trained together.

11/27, --identity: boxes of every tracked person in the five marked races,
sampled evenly, for a Teachable Machine model that tells racing skaters from
officials, coaches, staff and spectators. The race of each picture is kept,
so models can be trained leaving one group out (Beijing, Calgary, Inzell).
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
LATE_S = 3.0
MIN_LEG = 80
SKATERS = [("pair10_vanderpoel", "Bart Swings", "phase9/pose_pair10_vanderpoel_swings_v1.csv"),
           ("pair10_vanderpoel", "Nils van der Poel", "phase9/pose_pair10_vanderpoel_poel_v1.csv"),
           ("pair09_bloemen", "Ted-Jan Bloemen", "phase9/pose_pair09_bloemen_bloemen.csv"),
           ("pair06_bergsma", "Jorrit Bergsma", "phase9/pose_pair06_bergsma_bergsma.csv")]
MORE = {"pool": "phase9/tm_pool_more", "per_skater": 220, "near_s": 2.0}
IDENTITY = {"pool": "phase9/tm_id_pool", "out": "phase9/tm_id_labelled", "log": "phase9/tm_id_labels.csv",
            "pairs": {"pair10_vanderpoel": "beijing", "pair09_bloemen": "beijing", "pair06_bergsma": "beijing",
                      "calgary2025_ghiotto": "calgary", "inzell2026_eitrem": "inzell"},
            "per_pair": {"beijing": 110, "calgary": 260, "inzell": 260}, "min_box_h": 120}   # 720p pixels
UPPER = {"pool": "phase9/tm_upper_pool", "out": "phase9/tm_upper_labelled", "log": "phase9/tm_upper_labels.csv",
         "skaters": SKATERS + [("calgary2025_ghiotto", "Davide Ghiotto", "phase9/pose_calgary2025_ghiotto_ghiotto.csv")],
         "step": 3, "max_feet": 0.3, "min_box_h": 200}   # box height in 720p pixels (scaled per video)


def make_pool(upper=False, add=None):
    """add: (pair, skater, pose file) list appended to an existing pool instead of rebuilding it (11/27)."""
    pool_dir = UPPER["pool"] if upper else POOL
    os.makedirs(pool_dir, exist_ok=True)
    marks = pd.read_csv("phase9/phase_marks.csv")
    marks = marks[marks.use == "yes"]
    rows = []
    for pair, sk, pose in (add or (UPPER["skaters"] if upper else SKATERS)):
        m = marks[(marks.pair == pair) & (marks.skater == sk)]
        lab = frame_labels(m)
        d = pd.read_csv(pose)
        if upper:
            box_h = (d.box_y2 - d.box_y1) / race(pair)["scale"]
            d = d[d.track_id.notna() & (d.foot_vis < UPPER["max_feet"]) & (box_h >= UPPER["min_box_h"]) & d.frame.isin(lab.keys())]
            d = d.iloc[::UPPER["step"]].set_index("frame")
        else:
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
            cv2.imwrite(os.path.join(pool_dir, name), crop)
            rows.append({"image": name, "skater": sk, "frame": target, "suggested": lab[target],
                         "leg_px": round(x.leg_px) if pd.notna(x.leg_px) else 0})
        cap.release()
        print(sk, sum(r["skater"] == sk for r in rows), flush=True)
    p = pd.DataFrame(rows).sample(frac=1, random_state=21)    # mixed order, so one race does not come in a block
    if add:
        p = pd.concat([pd.read_csv(os.path.join(pool_dir, "pool.csv")), p], ignore_index=True)
    p.to_csv(os.path.join(pool_dir, "pool.csv"), index=False)
    print(len(p), "pictures;", p.suggested.value_counts().to_dict())


def make_more_pool():
    import phase_model as pm
    os.makedirs(MORE["pool"], exist_ok=True)
    done = set(pd.read_csv(LOG).image) if os.path.exists(LOG) else set()
    rows = []
    for pair, sk in pm.ROUNDS[4]["train"] + pm.ROUNDS[4]["test"]:
        d, marks = pm.load(pair, sk, 5)
        fps = race(pair)["fps"]
        edges = marks[marks.event.isin(["entry", "exit"])].frame.values
        d["near"] = d.frame.apply(lambda f: np.abs(edges - f).min() <= MORE["near_s"] * fps)
        half = MORE["per_skater"] // 2
        pick = pd.concat([g.sample(n=min(half, len(g)), random_state=27) for _, g in d.groupby("near")])
        pick = pick.sort_values("frame").set_index("frame")
        r = race(pair)
        cap = cv2.VideoCapture(r["video"])
        n = r["base"] - 1
        for target in pick.index:
            while n < target - 1:
                cap.grab(); n += 1
            ok, f = cap.read(); n += 1
            if not ok:
                break
            x = pick.loc[target]
            name = f"{pair}_{short(sk)}_{target}.jpg"
            if name in done:
                continue
            H, W = f.shape[:2]
            pw, ph = PAD * (x.box_x2 - x.box_x1), PAD * (x.box_y2 - x.box_y1)
            crop = f[int(max(0, x.box_y1 - ph)):int(min(H, x.box_y2 + ph)), int(max(0, x.box_x1 - pw)):int(min(W, x.box_x2 + pw))]
            cv2.imwrite(os.path.join(MORE["pool"], name), crop)
            rows.append({"image": name, "skater": sk, "frame": target, "suggested": x.label, "leg_px": round(x.leg_px),
                         "near_boundary": bool(x.near)})
        cap.release()
        print(sk, sum(r_["skater"] == sk for r_ in rows), flush=True)
    p = pd.DataFrame(rows).sample(frac=1, random_state=27)
    p.to_csv(os.path.join(MORE["pool"], "pool.csv"), index=False)
    print(len(p), "pictures;", p.suggested.value_counts().to_dict(), "; near a boundary:", int(p.near_boundary.sum()))


def make_identity_pool():
    from skater_pose import TRACKS
    os.makedirs(IDENTITY["pool"], exist_ok=True)
    rows = []
    for pair, group in IDENTITY["pairs"].items():
        r = race(pair)
        a, b = r["window"]
        t = pd.read_csv(TRACKS[pair])
        t = t[t.frame.between(a, b) & ((t.y2 - t.y1) / r["scale"] >= IDENTITY["min_box_h"])]
        t = t.sample(n=min(IDENTITY["per_pair"][group], len(t)), random_state=27).sort_values("frame")
        cap = cv2.VideoCapture(r["video"])
        n = r["base"] - 1
        for _, x in t.iterrows():
            while n < x.frame - 1:
                cap.grab(); n += 1
            if n < x.frame:
                ok, f = cap.read(); n += 1
                if not ok:
                    break
            H, W = f.shape[:2]
            pw, ph = PAD * (x.x2 - x.x1), PAD * (x.y2 - x.y1)
            crop = f[int(max(0, x.y1 - ph)):int(min(H, x.y2 + ph)), int(max(0, x.x1 - pw)):int(min(W, x.x2 + pw))]
            name = f"{pair}_id_{int(x.frame)}_{int(x.track_id)}.jpg"
            cv2.imwrite(os.path.join(IDENTITY["pool"], name), crop)
            rows.append({"image": name, "skater": group, "frame": int(x.frame), "suggested": "", "leg_px": 0, "pair": pair})
        cap.release()
        print(pair, sum(r_["pair"] == pair for r_ in rows), flush=True)
    p = pd.DataFrame(rows).sample(frac=1, random_state=27)
    p.to_csv(os.path.join(IDENTITY["pool"], "pool.csv"), index=False)
    print(len(p), "pictures")


def show(img, text, sub, done, total, keys="[C corner  S straight  X skip  U undo  Q quit]"):
    h, w = img.shape[:2]
    s = min(700 / h, 900 / w)
    big = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((800, 960, 3), 30, np.uint8)
    y0, x0 = 70 + (700 - big.shape[0]) // 2, (960 - big.shape[1]) // 2
    canvas[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
    cv2.putText(canvas, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, sub, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, f"{done} labelled of {total}   {keys}", (12, 790),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.imshow("label for Teachable Machine", canvas)


def late_only(pool):
    marks = pd.read_csv("phase9/phase_marks.csv")
    marks = marks[marks.use == "yes"]
    keep = []
    for pair, sk, _ in SKATERS:
        ex = marks[(marks.pair == pair) & (marks.skater == sk) & (marks.event == "exit")].frame.values
        fps = race(pair)["fps"]
        g = pool[(pool.skater == sk) & (pool.suggested == "straight")]
        since = g.frame.apply(lambda f: min([f - e for e in ex if e <= f], default=np.nan) / fps)
        keep.append(g[since >= LATE_S])
    return pd.concat(keep)


def label(late=False, upper=False, identity=False, more=False, first=None):
    global POOL, OUT, LOG
    if more:
        POOL = MORE["pool"]          # same label list and folders as the first full-body set
    if upper:
        POOL, OUT, LOG = UPPER["pool"], UPPER["out"], UPPER["log"]
    keymap = {"c": "corner", "s": "straight", "x": "skip"}
    keys = "[C corner  S straight  X skip  U undo  Q quit]"
    if identity:
        POOL, OUT, LOG = IDENTITY["pool"], IDENTITY["out"], IDENTITY["log"]
        keymap = {"r": "racer", "n": "not_racer", "x": "skip"}
        keys = "[R racer  N not a racer  X skip  U undo  Q quit]"
    pool = pd.read_csv(os.path.join(POOL, "pool.csv"))
    if late:
        pool = late_only(pool)
        print(len(pool), "late-straight pictures in the pool")
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=["image", "label"])
    done = list(log.itertuples(index=False, name=None))
    seen = {i for i, _ in done}
    todo = [r for r in pool.itertuples() if r.image not in seen]
    if first:
        # 11/27: pictures from these groups of races first, to even out the groups for training
        from tm_folds import group_of
        todo = sorted(todo, key=lambda r: group_of(r.image) not in first)
        print(sum(group_of(r.image) in first for r in todo), "pictures from", ", ".join(first), "come first")
    i = 0
    while i < len(todo):
        r = todo[i]
        img = cv2.imread(os.path.join(POOL, r.image))
        n_lab = sum(l != "skip" for _, l in done)
        if identity:
            show(img, "Is this a racing skater?", f"{r.pair}, frame {r.frame}", n_lab, len(pool), keys)
        else:
            show(img, f"{r.skater}", f"marks say: {r.suggested}   (leg {r.leg_px} px)", n_lab, len(pool), keys)
        k = cv2.waitKey(0) & 0xFF
        if k < 128 and chr(k).lower() in keymap:
            lab = keymap[chr(k).lower()]
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
    ap.add_argument("--late", action="store_true")
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--more", action="store_true")
    ap.add_argument("--first", help="groups to show first, for example calgary,inzell")
    ap.add_argument("--add", help="a marked pair to add to the upper-body pool, for example inzell2026_eitrem")
    args = ap.parse_args()
    first = args.first.split(",") if args.first else None
    if args.add:
        import phase_model as pm
        add = [(pair, sk, pm.DATA[(pair, sk)][0]) for pair, sk in pm.DATA if pair == args.add]
        make_pool(upper=True, add=add)
    elif args.more:
        make_more_pool() if args.pool else label(more=True, first=first)
    elif args.identity:
        make_identity_pool() if args.pool else label(identity=True)
    else:
        make_pool(args.upper) if args.pool else label(args.late, args.upper, first=first)


if __name__ == "__main__":
    main()
