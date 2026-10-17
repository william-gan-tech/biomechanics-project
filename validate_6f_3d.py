"""
Phase 6f (10/16): do MediaPipe's estimated 3D ("world") landmarks give angles
that depend less on camera view than the 2D image angles?

Target (fixed 10/15, before testing): 3D removes at least half of the
front-on vs side-on difference in trunk lean and knee angle, while side-on
values stay within ~5 deg of 2D.

Test 1 (main): recovered pieces (dual_tracking/recovered_segments.csv) of the
same skater and phase in the same race, filmed front-on and side-on. For each
skater x phase with both, gap = mean(front) - mean(side), in 2D and in 3D.
Test 2 (sanity): on side-on recovered pieces, 3D minus 2D per segment.

Angles compared like for like:
  trunk lean   2D torso_lean_angle_deg (right hip -> right shoulder, image plane)
               3D lean3d_right (same landmarks, angle from vertical in 3D)
  knee angle   2D right/left_knee_filtered, 3D right/left_knee3d (mean of both legs)

Usage:
    python -m validate_6f_3d
"""

import numpy as np
import pandas as pd

RECOVERED = "dual_tracking/recovered_segments.csv"
OUTPUT = "phase6f_3d_results.csv"
SHOTS = ("front", "side")


def segment_means():
    rec = pd.read_csv(RECOVERED, dtype=str).fillna("")
    rec = rec[rec.shot.isin(SHOTS)]
    poses = {}
    rows = []
    for r in rec.itertuples():
        if r.pose_source not in poses:
            poses[r.pose_source] = pd.read_csv(r.pose_source)
        p = poses[r.pose_source]
        f = p[(p.piece_no == int(r.piece_no)) & (p.frame >= int(r.start_frame)) & (p.frame <= int(r.end_frame))]
        f = f.dropna(subset=["lean3d_right", "right_knee3d", "left_knee3d"])
        if len(f) < 10:
            continue
        rows.append({
            "skater": r.skater, "phase": r.phase, "shot": r.shot, "piece": r.piece_no, "frames": len(f),
            "lean_2d": f.torso_lean_angle_deg.mean(), "lean_3d": f.lean3d_right.mean(),
            "knee_2d": ((f.right_knee_filtered + f.left_knee_filtered) / 2).mean(),
            "knee_3d": ((f.right_knee3d + f.left_knee3d) / 2).mean(),
        })
    return pd.DataFrame(rows)


def main():
    seg = segment_means()
    pd.set_option("display.width", 200)
    print(f"{len(seg)} recovered segments with 3D angles "
          f"({(seg.shot == 'front').sum()} front-on, {(seg.shot == 'side').sum()} side-on)\n")

    # Test 1: front vs side gap per skater x phase
    gaps = []
    for (sk, ph), g in seg.groupby(["skater", "phase"]):
        fr, sd = g[g.shot == "front"], g[g.shot == "side"]
        if fr.empty or sd.empty:
            continue
        row = {"skater": sk, "phase": ph, "n_front": len(fr), "n_side": len(sd)}
        for m in ("lean", "knee"):
            row[f"{m}_gap_2d"] = fr[f"{m}_2d"].mean() - sd[f"{m}_2d"].mean()
            row[f"{m}_gap_3d"] = fr[f"{m}_3d"].mean() - sd[f"{m}_3d"].mean()
        gaps.append(row)
    gaps = pd.DataFrame(gaps)
    print("TEST 1: front-on minus side-on, same skater and phase")
    print(gaps.round(1).to_string(index=False))
    verdict = {}
    for m in ("lean", "knee"):
        g2, g3 = gaps[f"{m}_gap_2d"].abs().mean(), gaps[f"{m}_gap_3d"].abs().mean()
        removed = 1 - g3 / g2 if g2 else np.nan
        verdict[m] = removed >= 0.5
        print(f"  {m}: mean |gap| 2D {g2:.1f} deg -> 3D {g3:.1f} deg  ({removed:.0%} of the gap removed; "
              f"target >= 50%: {'MET' if verdict[m] else 'NOT MET'})")

    # Test 2: side-on agreement
    side = seg[seg.shot == "side"]
    d_lean, d_knee = (side.lean_3d - side.lean_2d), (side.knee_3d - side.knee_2d)
    print(f"\nTEST 2: side-on segments (n = {len(side)}), 3D minus 2D")
    print(f"  trunk lean: mean {d_lean.mean():+.1f} deg, median |diff| {d_lean.abs().median():.1f} deg")
    print(f"  knee angle: mean {d_knee.mean():+.1f} deg, median |diff| {d_knee.abs().median():.1f} deg")
    side_ok = d_lean.abs().median() <= 5 and d_knee.abs().median() <= 5
    print(f"  within ~5 deg: {'YES' if side_ok else 'NO'}")

    print(f"\n6f TARGET: {'MET' if all(verdict.values()) and side_ok else 'NOT MET'}")
    seg.to_csv(OUTPUT, index=False)
    print(f"Saved per-segment values -> {OUTPUT}")


if __name__ == "__main__":
    main()
