"""
Phase 6a: score the dual tracker against the researcher's identity check.

Ground truth (dual_tracking/identity_ground_truth.csv) comes from the
researcher's review of the sampled frames (every 6th frame) on each clip's
review sheet: which track ID was on skater A and on skater B, frame by frame.
Frames where a skater had no box are simply absent.

Scoring, per skater per sampled frame where the skater had a box:
  - correct:       the skater has exactly one box, not shared with the other
                   skater, and its ID was not last used on the other skater
  - swapped:       the ID was last on the OTHER skater (identity contaminated:
                   that track's data now mixes two people)
  - merged:        one box covers both skaters
  - duplicate:     the skater has two boxes in the same frame
A NEW ID on the same skater is not an error here (the track just restarts; it
shortens segments but doesn't mix people) -- it is counted separately.

Headline: identity correct = correct / all skater-frames with a box.
Target fixed 10/8, before testing: >= 95%.

Usage:
    python -m score_dual_tracking
"""

import pandas as pd

GT = "dual_tracking/identity_ground_truth.csv"
STEP = 6
TARGET = 0.95
CLIP_STARTS = {  # first frame of each clip (sampling grid = start + k*STEP)
    "rijhnen_lehman_beijing": 28260, "bergsma_ichinohe_beijing": 126270,
    "bloemen_zakharov_beijing_a": 167184, "bloemen_zakharov_beijing_b": 172374,
    "eitrem_jilek_inzell": 6428, "ghiotto_eitrem_start": 0,
}


def expand(gt):
    rows = []
    for _, r in gt.iterrows():
        start = CLIP_STARTS[r["clip"]]
        f = int(r["first_frame"])
        while f <= int(r["last_frame"]):
            if (f - start) % STEP == 0:
                rows.append({"clip": r["clip"], "skater": r["skater"], "frame": f, "track_id": int(r["track_id"])})
            f += 1
    return pd.DataFrame(rows).drop_duplicates()


def score_clip(df):
    owner, new_ids = {}, 0
    tallies = {"correct": 0, "swapped": 0, "merged": 0, "duplicate": 0}
    swaps = 0
    for frame, g in df.groupby("frame"):
        ids_by_skater = g.groupby("skater").track_id.apply(set).to_dict()
        for skater, ids in ids_by_skater.items():
            other = "B" if skater == "A" else "A"
            other_ids = ids_by_skater.get(other, set())
            if len(ids) > 1:
                tallies["duplicate"] += 1
            elif ids & other_ids:
                tallies["merged"] += 1
            else:
                tid = next(iter(ids))
                if owner.get(tid) == other:
                    tallies["swapped"] += 1
                    swaps += 1 if owner.get(("last", skater)) != tid else 0
                else:
                    tallies["correct"] += 1
                    if tid not in owner:
                        new_ids += 1
            for tid in ids:
                if not (ids & other_ids):
                    owner[tid] = skater
                owner[("last", skater)] = tid
    total = sum(tallies.values())
    return {**tallies, "skater_frames": total, "swap_events": swaps, "ids_used": new_ids,
            "identity_correct": tallies["correct"] / total if total else float("nan")}


def score_clip_strict(df):
    """Stricter, standard-style rule (10/9): each track ID belongs to the skater
    it covers in most frames; every frame where that ID is on the OTHER skater
    is wrong, because the track's data mixes two people. (score_clip only
    counted the frame where a swap happened.) Merged and duplicate boxes are
    wrong as before."""
    counts = df.groupby(["track_id", "skater"]).frame.nunique().unstack(fill_value=0)
    owner = counts.idxmax(axis=1).to_dict()
    correct = mixed = merged = dup = 0
    for frame, g in df.groupby("frame"):
        ids_by_skater = g.groupby("skater").track_id.apply(set).to_dict()
        for skater, ids in ids_by_skater.items():
            other_ids = ids_by_skater.get("B" if skater == "A" else "A", set())
            if len(ids) > 1:
                dup += 1
            elif ids & other_ids:
                merged += 1
            elif owner[next(iter(ids))] != skater:
                mixed += 1
            else:
                correct += 1
    total = correct + mixed + merged + dup
    return {"correct": correct, "on_wrong_track": mixed, "merged": merged, "duplicate": dup,
            "skater_frames": total, "identity_correct": correct / total if total else float("nan")}


def score_labels(df_gt, labels, init_frames):
    """Score relabel_dual.py output (A/B per box) against the ground truth.
    Per ground-truth skater-frame (excluding the one initialisation frame per
    skater): correct if that skater's box is labelled as them; wrong if
    labelled as the other skater; missed if unlabelled. Where the ground truth
    has one box around both skaters, correct = left unlabelled. Also counts
    false positives: boxes labelled A/B at checked frames that are not a
    skater box at all (officials, crowd)."""
    lab = {(int(r.frame), int(r.track_id)): r.label if isinstance(r.label, str) else ""
           for r in labels.itertuples()}
    correct = wrong = missed = 0
    for frame, g in df_gt.groupby("frame"):
        ids_by_skater = g.groupby("skater").track_id.apply(set).to_dict()
        for skater, ids in ids_by_skater.items():
            if init_frames.get(skater) == frame:
                continue
            other = "B" if skater == "A" else "A"
            if ids & ids_by_skater.get(other, set()):  # one box around both
                if all(lab.get((frame, t), "") == "" for t in ids):
                    correct += 1
                else:
                    wrong += 1
                continue
            got = [lab.get((frame, t), "") for t in ids]
            if other in got:
                wrong += 1
            elif skater in got:
                correct += 1
            else:
                missed += 1
    checked = set(df_gt.frame)
    gt_boxes = set(zip(df_gt.frame, df_gt.track_id))
    fp = sum(1 for (f, t), l in lab.items() if l and f in checked and (f, t) not in gt_boxes)
    total = correct + wrong + missed
    return {"correct": correct, "wrong_skater": wrong, "missed": missed, "skater_frames": total,
            "false_positives": fp, "identity_correct": correct / total if total else float("nan")}


def main_labels():
    import os
    gt_raw = pd.read_csv(GT)
    df = expand(gt_raw)
    rows = []
    for clip, g in df.groupby("clip", sort=False):
        path = f"dual_tracking/{clip}_labels.csv"
        if not os.path.exists(path):
            continue
        labels = pd.read_csv(path)
        # the one-time identification frames used by relabel_dual.init_refs
        init = {}
        for skater in ("A", "B"):
            mine = g[g.skater == skater].sort_values("frame")
            other = g[g.skater != skater]
            for r in mine.itertuples():
                if not ((other.frame == r.frame) & (other.track_id == r.track_id)).any():
                    init[skater] = r.frame
                    break
        rows.append({"clip": clip, **score_labels(g, labels, init)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("APPEARANCE-BASED A/B LABELS (relabel_dual.py) vs researcher ground truth\n")
    print(res.round(3).to_string(index=False))
    tot = res[["correct", "wrong_skater", "missed", "skater_frames", "false_positives"]].sum()
    overall = tot.correct / tot.skater_frames
    wrong_rate = tot.wrong_skater / tot.skater_frames
    print(f"\nOverall: correct {overall:.1%}, on the WRONG skater {wrong_rate:.1%}, "
          f"missed {tot.missed / tot.skater_frames:.1%} of {int(tot.skater_frames)} skater-frames; "
          f"false positives {int(tot.false_positives)}")
    print(f"Target >= {TARGET:.0%} correct: {'MET' if overall >= TARGET else 'NOT MET'}")
    res.to_csv("dual_tracking/identity_scores_relabel.csv", index=False)


JUMP_FACTOR = 1.0     # cut a track if its box centre moves more than this x box width between frames
OVERLAP_IOU = 0.05    # two skater-sized boxes overlapping this much = unsafe frame
OVERLAP_PAD = 3       # frames dropped either side of an unsafe frame
SKATER_SIZE = 0.5     # skater-sized = box height >= this x the largest box in the frame
MIN_PIECE = 12        # pieces shorter than this (~0.5 s) are discarded


def pure_pieces(tracks):
    """10/9: split the tracker's tracks into pieces that should each contain ONE
    person, using two rules that don't depend on appearance:
      - a box can't teleport: cut where a track's box centre jumps more than
        JUMP_FACTOR x its width between consecutive frames (catches an ID
        being handed to the other skater, as in Bloemen/Zakharov 'a');
      - drop frames near any overlap between two skater-sized boxes (where
        merged boxes and swaps happen), cutting tracks there.
    Returns {(frame, track_id): piece_id} for frames kept."""
    t = tracks.copy()
    t["h"] = t.y2 - t.y1
    t["w"] = t.x2 - t.x1
    t["cx"] = (t.x1 + t.x2) / 2
    t["h_rel"] = t.h / t.groupby("frame").h.transform("max")
    big = t[t.h_rel >= SKATER_SIZE]

    unsafe = set()
    for f, g in big.groupby("frame"):
        b = g[["x1", "y1", "x2", "y2"]].values
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                ix = max(0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
                iy = max(0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
                inter = ix * iy
                union = ((b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]) + (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1]) - inter)
                if union > 0 and inter / union >= OVERLAP_IOU:
                    unsafe.update(range(int(f) - OVERLAP_PAD, int(f) + OVERLAP_PAD + 1))

    pieces, pid = {}, 0
    for tid, g in big.sort_values("frame").groupby("track_id"):
        current, prev = [], None
        for r in g.itertuples():
            f = int(r.frame)
            cut = (f in unsafe or prev is None or f - prev.frame > 1
                   or abs(r.cx - prev.cx) > JUMP_FACTOR * max(r.w, prev.w))
            if cut and current:
                if len(current) >= MIN_PIECE:
                    for k in current:
                        pieces[k] = pid
                    pid += 1
                current = []
            if f not in unsafe:
                current.append((f, int(tid)))
            prev = r
        if len(current) >= MIN_PIECE:
            for k in current:
                pieces[k] = pid
            pid += 1
    return pieces


def main_purity():
    """Score pure_pieces against the ground truth: a kept piece is 'mixed' if
    it contains checked frames of both skaters (or a box around both).
    Coverage = share of checked skater-frames that fall inside kept pieces."""
    df = expand(pd.read_csv(GT))
    rows = []
    for clip, g in df.groupby("clip", sort=False):
        tracks = pd.read_csv(f"dual_tracking/{clip}_tracks.csv")
        pieces = pure_pieces(tracks)
        gt = {}
        for r in g.itertuples():
            gt.setdefault((int(r.frame), int(r.track_id)), set()).add(r.skater)
        members = {}
        for key, skaters in gt.items():
            if key in pieces:
                members.setdefault(pieces[key], []).append(skaters)
        mixed_frames = kept = 0
        for pid, lst in members.items():
            flat = [s for sk in lst for s in sk]
            maj = max(set(flat), key=flat.count)
            for sk in lst:
                kept += len(sk)
                if len(sk) > 1 or maj not in sk:
                    mixed_frames += len(sk)
        total = sum(len(s) for s in gt.values())
        rows.append({"clip": clip, "pieces_checked": len(members), "skater_frames": total,
                     "kept": kept, "coverage": kept / total if total else float("nan"),
                     "on_wrong_skater_in_kept": mixed_frames,
                     "purity": 1 - mixed_frames / kept if kept else float("nan")})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("PURE PIECES (tracker IDs, cut at box jumps and skater overlaps)\n")
    print(res.round(3).to_string(index=False))
    tot = res[["skater_frames", "kept", "on_wrong_skater_in_kept"]].sum()
    print(f"\nOverall: purity {1 - tot.on_wrong_skater_in_kept / tot.kept:.1%} of kept skater-frames are on the "
          f"right skater; coverage {tot.kept / tot.skater_frames:.1%} of checked skater-frames kept")
    res.to_csv("dual_tracking/identity_scores_pure_pieces.csv", index=False)


def main():
    import sys
    if "--labels" in sys.argv:
        return main_labels()
    if "--purity" in sys.argv:
        return main_purity()
    df = expand(pd.read_csv(GT))
    rows = [{"clip": c, **score_clip_strict(g)} for c, g in df.groupby("clip", sort=False)]
    print("STRICT RULE (headline): each ID belongs to the skater it covers most; frames on the other skater are wrong\n")
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    tot = res[["correct", "on_wrong_track", "merged", "duplicate", "skater_frames"]].sum()
    overall = tot.correct / tot.skater_frames
    print(f"\nOverall identity correct: {overall:.1%} of {int(tot.skater_frames)} checked skater-frames "
          f"(on wrong skater's track {int(tot.on_wrong_track)}, merged {int(tot.merged)}, duplicate {int(tot.duplicate)})")
    lenient = pd.DataFrame([{"clip": c, **score_clip(g)} for c, g in df.groupby("clip", sort=False)])
    lt = lenient[["correct", "skater_frames"]].sum()
    print(f"(Lenient rule, counting only the frame of each swap: {lt.correct / lt.skater_frames:.1%} -- "
          f"not used as the headline because it hides tracks that mix two skaters.)")
    print(f"Target >= {TARGET:.0%}: {'MET' if overall >= TARGET else 'NOT MET'}")
    res.to_csv("dual_tracking/identity_scores.csv", index=False)


if __name__ == "__main__":
    main()
