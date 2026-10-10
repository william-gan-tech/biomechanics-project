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


def main():
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
