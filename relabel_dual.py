"""
Phase 6b (10/9): assign tracked boxes to skater A / skater B by suit
appearance, instead of trusting the tracker's ID numbers.

Why: the 6a identity check (89.9%, target 95%) found every identity error
comes from (1) the two skaters overlapping -- one box around both, IDs coming
back swapped -- and (2) one skater briefly losing their box, with their ID
then handed to the other skater. Both are failures of the tracker's ID
numbers, not of detection. But a long-track pair is always exactly TWO
skaters in clearly different suits, so appearance can decide who is who.

Method, per clip:
  1. One-time identification: the first frame where each skater has their
     own box gives a reference suit appearance (colour histogram of the torso
     area). In real use this is a single click per skater; for testing it is
     taken from the researcher's ground truth for that ONE frame per skater,
     and every other frame is scored.
  2. Every frame: compare each box's appearance with A and B, and assign the
     best one-to-one match (Hungarian), accepting a match only if it is
     clearly closer to that skater than to the other (MARGIN).
  3. Merged boxes: if a frame's boxes give only one match while that box is
     unusually wide (both skaters inside), it is dropped rather than guessed.
  4. Smoothing: within each tracker ID, isolated single-frame flips are
     overruled by the neighbouring frames (the tracker's IDs are reliable
     over short spans, just not across overlaps).
  5. References are slowly updated with confident matches, to follow
     lighting changes across a clip.

Output: dual_tracking/<clip>_labels.csv (frame, track_id, label A/B/'' ,
similarities). Scored by score_dual_tracking.py (--labels).

Usage:
    python -m relabel_dual
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

OUT_DIR = "dual_tracking"
CLIPS = "dual_tracking/test_clips.csv"
GT = "dual_tracking/identity_ground_truth.csv"
MARGIN = 0.10        # accept a box as skater X only if sim(X) - sim(other) >= MARGIN
MIN_SIM = 0.20       # ... and sim(X) >= MIN_SIM
WIDE_FACTOR = 1.6    # a lone matched box this much wider than the skater's usual box = merged
UPDATE_RATE = 0.05   # reference update weight for confident matches


def torso_hist(frame, x1, y1, x2, y2):
    h, w = y2 - y1, x2 - x1
    tx1, tx2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    ty1, ty2 = int(y1 + 0.15 * h), int(y1 + 0.6 * h)
    H, W = frame.shape[:2]
    tx1, ty1, tx2, ty2 = max(0, tx1), max(0, ty1), min(W, tx2), min(H, ty2)
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None
    hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def sim(a, b):
    return float(cv2.compareHist(a.reshape(30, 32), b.reshape(30, 32), cv2.HISTCMP_CORREL))


def box_hists(video, start, end, tracks):
    """Appearance histogram for every tracked box, reading frames in order."""
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = {}
    n = start
    while n <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if n in by_frame:
            for _, r in by_frame[n].iterrows():
                hst = torso_hist(frame, r.x1, r.y1, r.x2, r.y2)
                if hst is not None:
                    out[(n, int(r.track_id))] = hst
        n += 1
    cap.release()
    return out


def init_refs(gt_clip, hists):
    """Reference appearance for A and B from the first frame where each has
    its own box (not shared with the other skater)."""
    refs, widths_frame = {}, {}
    for skater in ("A", "B"):
        rows = gt_clip[gt_clip.skater == skater].sort_values("first_frame")
        other = gt_clip[gt_clip.skater != skater]
        for _, r in rows.iterrows():
            shared = ((other.track_id == r.track_id) & (other.first_frame <= r.first_frame)
                      & (other.last_frame >= r.first_frame)).any()
            key = (int(r.first_frame), int(r.track_id))
            if not shared and key in hists:
                refs[skater] = hists[key].copy()
                widths_frame[skater] = key
                break
    return refs, widths_frame


def relabel_clip(name, video, start, end, gt_clip):
    tracks = pd.read_csv(os.path.join(OUT_DIR, f"{name}_tracks.csv"))
    hists = box_hists(video, start, end, tracks)
    refs, init_keys = init_refs(gt_clip, hists)
    if set(refs) != {"A", "B"}:
        return None
    usual_w = {}
    for s, (f, tid) in init_keys.items():
        r = tracks[(tracks.frame == f) & (tracks.track_id == tid)].iloc[0]
        usual_w[s] = r.x2 - r.x1

    rows = []
    for f, g in tracks.groupby("frame"):
        g = g[[(f, int(t)) in hists for t in g.track_id]]
        if g.empty:
            continue
        H = np.array([hists[(f, int(t))] for t in g.track_id])
        S = np.array([[sim(h, refs["A"]), sim(h, refs["B"])] for h in H])
        cost = -S
        ri, ci = linear_sum_assignment(cost)
        labels = {i: "" for i in range(len(g))}
        for i, c in zip(ri, ci):
            mine, theirs = S[i, c], S[i, 1 - c]
            if mine >= MIN_SIM and mine - theirs >= MARGIN:
                labels[i] = "AB"[c]
        matched = [i for i, l in labels.items() if l]
        if len(matched) == 1:
            i = matched[0]
            w = g.iloc[i].x2 - g.iloc[i].x1
            if w > WIDE_FACTOR * usual_w[labels[i]]:
                labels[i] = ""  # likely one box around both skaters
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append({"frame": f, "track_id": int(r.track_id), "label": labels[i],
                         "sim_A": S[i, 0], "sim_B": S[i, 1], "width": r.x2 - r.x1})
            if labels[i] and abs(S[i, 0] - S[i, 1]) >= 2 * MARGIN:
                refs[labels[i]] = (1 - UPDATE_RATE) * refs[labels[i]] + UPDATE_RATE * H[i]

    df = pd.DataFrame(rows).sort_values(["track_id", "frame"])
    # smoothing: a label that differs from both neighbours within the same track is overruled
    smoothed = []
    for tid, g in df.groupby("track_id"):
        lab = g.label.tolist()
        for k in range(1, len(lab) - 1):
            if lab[k - 1] and lab[k - 1] == lab[k + 1] and lab[k] != lab[k - 1]:
                lab[k] = lab[k - 1]
        smoothed.append(g.assign(label=lab))
    df = pd.concat(smoothed).sort_values(["frame", "track_id"])
    df.to_csv(os.path.join(OUT_DIR, f"{name}_labels.csv"), index=False)
    return df, init_keys


FLIP_RUN = 6         # consecutive frames of the opposite appearance needed to cut a track (a swap)
PIECE_MARGIN = 0.05  # a track piece is labelled only if its median sim difference exceeds this
MIN_HEIGHT_FRAC = 0.5  # piece's median box height vs the skater's usual height (drops background people)
REF_FRAMES = 10      # frames of the identification track used for the reference appearance


def relabel_clip_hybrid(name, video, start, end, gt_clip):
    """v2 (10/9): v1 matched every frame independently and did WORSE than the
    tracker's own IDs (72.6% vs 89.9%) -- per-frame appearance is noisy, similar
    suits (Rijhnen/Lehman, both mostly black) get confused, and same-colour
    officials (Dutch coaches in orange) get labelled. So: keep the tracker's
    tracks (reliable most of the time), cut a track only where its appearance
    persistently flips to the other skater (a swap), and label each piece by its
    MEDIAN appearance over many frames."""
    tracks = pd.read_csv(os.path.join(OUT_DIR, f"{name}_tracks.csv"))
    hists = box_hists(video, start, end, tracks)
    _, init_keys = init_refs(gt_clip, hists)
    if set(init_keys) != {"A", "B"}:
        return None
    refs, usual_h = {}, {}
    for s, (f0, tid) in init_keys.items():
        t = tracks[(tracks.track_id == tid) & (tracks.frame >= f0)].sort_values("frame").head(REF_FRAMES)
        hs = [hists[(int(r.frame), tid)] for r in t.itertuples() if (int(r.frame), tid) in hists]
        refs[s] = np.mean(hs, axis=0)
        usual_h[s] = float((t.y2 - t.y1).median())
    min_h = MIN_HEIGHT_FRAC * min(usual_h.values())

    tracks = tracks[[(int(f), int(t)) in hists for f, t in zip(tracks.frame, tracks.track_id)]].copy()
    tracks["d"] = [sim(hists[(int(f), int(t))], refs["A"]) - sim(hists[(int(f), int(t))], refs["B"])
                   for f, t in zip(tracks.frame, tracks.track_id)]
    tracks["h"] = tracks.y2 - tracks.y1
    # v3: size relative to the LARGEST box in the same frame -- the skaters are
    # what the camera follows; background officials/coaches are smaller. (v2
    # used the skater's size at identification, which was far too lenient when
    # identification happened while the skater was far away.)
    tracks["h_rel"] = tracks.h / tracks.groupby("frame").h.transform("max")
    init_tracks = {tid: s for s, (f0, tid) in init_keys.items()}

    pieces = []
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        # v3: only a CLEAR appearance difference counts toward a cut; near-zero
        # differences (similar suits, e.g. Rijhnen/Lehman) are neutral, so noise
        # can't split a track that the tracker had right
        d = g.d.values
        sign = np.where(np.abs(d) >= PIECE_MARGIN, np.sign(d), 0)
        cut_points, run_start = [0], 0
        current = next((s for s in sign if s != 0), 0)  # the piece's established side
        for k in range(1, len(sign)):
            if sign[k] != sign[run_start]:
                run_start = k
            if (sign[run_start] != 0 and sign[run_start] != current
                    and k - run_start + 1 == FLIP_RUN and run_start > cut_points[-1]):
                cut_points.append(run_start)
                current = sign[run_start]
        cut_points.append(len(g))
        for k, (a, b) in enumerate(zip(cut_points[:-1], cut_points[1:])):
            p = g.iloc[a:b]
            med = float(p.d.median())
            label, strength = "", abs(med)
            if p.h_rel.median() < MIN_HEIGHT_FRAC:
                pass  # background person
            elif k == 0 and tid in init_tracks:
                # v3: the identification track keeps its skater until appearance
                # cuts it (the tracker's IDs are right most of the time)
                label, strength = init_tracks[tid], max(strength, 1.0)
            elif abs(med) >= PIECE_MARGIN:
                label = "A" if med > 0 else "B"
            pieces.append((tid, int(p.frame.min()), int(p.frame.max()), label, strength, len(p)))

    # one A and one B per frame: if two pieces claim the same skater, keep the stronger
    assign = {}
    for tid, f0, f1, label, strength, n in sorted(pieces, key=lambda x: -x[4] * np.sqrt(x[5])):
        for f in tracks[(tracks.track_id == tid) & (tracks.frame.between(f0, f1))].frame:
            key = (int(f), int(tid))
            taken = any(v == label for (ff, _), v in assign.items() if ff == int(f)) if label else False
            assign[key] = "" if taken else label

    # merged boxes: a frame with only one labelled skater whose box is unusually wide
    out = tracks[["frame", "track_id", "d", "h"]].copy()
    out["width"] = tracks.x2 - tracks.x1
    out["label"] = [assign.get((int(f), int(t)), "") for f, t in zip(out.frame, out.track_id)]
    usual_w = {}
    for s, (f0, tid) in init_keys.items():
        r = tracks[(tracks.frame == f0) & (tracks.track_id == tid)]
        usual_w[s] = float((r.x2 - r.x1).iloc[0]) if not r.empty else np.inf
    for f, g in out.groupby("frame"):
        lab = g[g.label != ""]
        if len(lab) == 1 and lab.width.iloc[0] > WIDE_FACTOR * usual_w[lab.label.iloc[0]]:
            out.loc[lab.index, "label"] = ""
    out = out.sort_values(["frame", "track_id"])
    out.to_csv(os.path.join(OUT_DIR, f"{name}_labels.csv"), index=False)
    return out, init_keys


def main():
    import sys
    clips = pd.read_csv(CLIPS, encoding="utf-8-sig")
    gt = pd.read_csv(GT)
    fn = relabel_clip if "--per-frame" in sys.argv else relabel_clip_hybrid
    for _, c in clips.iterrows():
        res = fn(c["name"], c["video"], int(c["start"]), int(c["end"]), gt[gt["clip"] == c["name"]])
        if res is None:
            print(f"{c['name']:28s} could not initialise both skaters")
            continue
        df, keys = res
        print(f"{c['name']:28s} init frames A={keys['A'][0]} B={keys['B'][0]}  "
              f"boxes labelled A={int((df.label == 'A').sum())} B={int((df.label == 'B').sum())} "
              f"unassigned={int((df.label == '').sum())}", flush=True)


if __name__ == "__main__":
    main()
