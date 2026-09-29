"""
Phase 5d: Stride Rhythm & Cadence Consistency.

Measures how evenly-spaced strides are -- distinct from the single-frame
velocity/lean features. Reuses the pipeline's existing stride detection
(peaks in knee angle, as in pipeline_engine.py), with two changes:

  - Minimum peak spacing is set in SECONDS, not frames, so 25 fps and
    29.97 fps videos are treated the same (see the 9/28 fps fix in
    compare_to_elite_reference.py).
  - Frames dropped by filter_implausible_frames leave gaps; short gaps are
    interpolated so they don't masquerade as long stride intervals. Segments
    with larger gaps are reported, not silently used.

Per segment:
  - stride_period_s   mean time between successive same-leg knee-flexion
                      troughs (one full stride cycle of that leg); intervals
                      spanning a tracking gap are excluded
  - cadence_per_min   60 / stride_period_s
  - rhythm_cv         std / mean of the peak-to-peak intervals.
                      Lower = more even rhythm. This is the core 5d metric.
  - lr_phase_offset   when the LEFT knee peaks, as a fraction of the right
                      leg's cycle. 0.5 = perfectly alternating legs; far from
                      0.5 = timing asymmetry between legs (ties into 5e).

Every detection is also plotted so peaks can be checked by eye against the
signal -- per the project rule of never trusting an algorithmic detection
without looking at it.

Usage:
    python -m stride_rhythm
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_to_elite_reference import load_labeled_segments, get_segment_frames, get_video_fps

MOVING_PHASES = ["start_acceleration", "corner", "straightaway"]
MIN_STRIDE_SECONDS = 0.35   # no real skating stride cycle is shorter than this
PEAK_PROMINENCE_DEG = 5.0   # floor; same prominence app.py uses for knee-angle strides
REL_PROMINENCE = 0.25       # trough must be >= 25% of the segment's 5-95% knee-angle range
# Tuning this by eye on 8 segments showed no single value works for all of
# them (9/28: 0.4 fixed start_candidate_3 785-884's shallow false troughs but
# broke Silovs corner and start_candidate_2). So every segment is also run at
# each of these, and rhythm_cv is only marked STABLE if it barely moves --
# a result that depends on the threshold is a detection artifact, not rhythm.
SENSITIVITY_REL = [0.0, 0.25, 0.4]
STABLE_CV_RANGE = 0.10
MAX_INTERP_GAP_FRAMES = 3
MIN_PEAKS = 3               # need at least 2 intervals to measure rhythm
OUTPUT_PATH = "phase5d_stride_rhythm_results.csv"
PLOT_DIR = "stride_rhythm_checks"


def continuous_signal(segment, col):
    """Reindex onto every frame in the segment and bridge short gaps. Returns
    (frames, values, n_unbridged_gap_frames)."""
    s = segment.set_index("frame")[col]
    s = s[~s.index.duplicated()]
    full = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
    bridged = full.interpolate(limit=MAX_INTERP_GAP_FRAMES, limit_area="inside")
    return bridged.index.values, bridged.values, int(bridged.isna().sum())


def detect_peaks(values, fps, rel_prominence=None):
    """Stride events = knee-angle TROUGHS (maximum flexion). Checked by eye
    9/28: extension peaks sit on flat 160-190 deg plateaus, so their timing
    jitters by several frames; the flexion troughs are sharp and regular.
    Some troughs read implausibly low (12-30 deg), likely leg-crossing
    occlusion in side view -- their TIMING still tracks the stride cycle,
    but the trough angle itself shouldn't be used as a measurement."""
    valid = ~np.isnan(values)
    filled = np.where(valid, values, np.nanmedian(values))
    # A fixed 5 deg prominence also caught shallow 10-20 deg dips between real
    # strides (seen by eye in start_candidate_3 785-884), so the required
    # depth scales with this segment's own knee-angle range.
    p5, p95 = np.nanpercentile(values, [5, 95])
    rel = REL_PROMINENCE if rel_prominence is None else rel_prominence
    prominence = max(PEAK_PROMINENCE_DEG, rel * (p95 - p5))
    peaks, _ = find_peaks(-filled, distance=max(1, int(MIN_STRIDE_SECONDS * fps)),
                          prominence=prominence)
    return peaks[valid[peaks]]


def clean_intervals(peaks, values, fps):
    """Peak-to-peak intervals, EXCLUDING any interval that spans an unbridged
    tracking gap (those measure the gap, not the stride)."""
    gap = np.isnan(values)
    keep = [not gap[a:b].any() for a, b in zip(peaks[:-1], peaks[1:])]
    return (np.diff(peaks) / fps)[keep], np.array(keep, dtype=bool)


def analyze_segment(row):
    seg = get_segment_frames(row)
    if seg is None:
        return None, "no data"
    fps = get_video_fps(row["video_path"])
    if fps is None:
        return None, "fps unreadable"

    frames, right, gaps_r = continuous_signal(seg, "right_knee_filtered")
    _, left, gaps_l = continuous_signal(seg, "left_knee_filtered")
    duration_s = len(frames) / fps

    peaks_r = detect_peaks(right, fps)
    peaks_l = detect_peaks(left, fps)

    sens_cv = []
    for rel in SENSITIVITY_REL:
        iv, _ = clean_intervals(detect_peaks(right, fps, rel), right, fps)
        sens_cv.append(iv.std(ddof=1) / iv.mean() if len(iv) >= MIN_PEAKS - 1 else np.nan)

    result = {
        "skater": row["skater"], "phase": row["phase"],
        "segment": f"{row['start_frame']}-{row['end_frame']}", "fps": round(fps, 2),
        "duration_s": round(duration_s, 2), "unbridged_gap_frames": max(gaps_r, gaps_l),
        "n_peaks_right": len(peaks_r), "n_peaks_left": len(peaks_l),
        "cv_sensitivity": "/".join("--" if np.isnan(c) else f"{c:.2f}" for c in sens_cv),
        "stable": bool(not np.isnan(sens_cv).any() and np.ptp(sens_cv) <= STABLE_CV_RANGE),
    }
    _plot(row, frames, right, left, peaks_r, peaks_l)

    intervals, keep = clean_intervals(peaks_r, right, fps)
    result["n_intervals"] = len(intervals)
    if len(intervals) < MIN_PEAKS - 1:
        return result, (f"only {len(intervals)} gap-free stride intervals in {duration_s:.1f}s "
                        f"(need {MIN_PEAKS - 1})")

    result.update({
        "stride_period_s": intervals.mean(),
        "cadence_per_min": 60.0 / intervals.mean(),
        "rhythm_cv": intervals.std(ddof=1) / intervals.mean(),
    })

    # Left-leg timing within each (gap-free) right-leg cycle
    offsets = []
    cycles = [(a, b) for (a, b), k in zip(zip(peaks_r[:-1], peaks_r[1:]), keep) if k]
    for a, b in cycles:
        inside = peaks_l[(peaks_l > a) & (peaks_l < b)]
        if len(inside) == 1:
            offsets.append((inside[0] - a) / (b - a))
    result["lr_phase_offset"] = float(np.mean(offsets)) if offsets else np.nan
    result["lr_cycles_used"] = len(offsets)
    return result, None


def _plot(row, frames, right, left, peaks_r, peaks_l):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frames, right, label="right knee", color="#1f77b4")
    ax.plot(frames, left, label="left knee", color="#ff7f0e", alpha=0.7)
    ax.plot(frames[peaks_r], right[peaks_r], "v", color="#1f77b4", markersize=8)
    ax.plot(frames[peaks_l], left[peaks_l], "v", color="#ff7f0e", markersize=8)
    ax.set_title(f"{row['skater']} | {row['phase']} | frames {row['start_frame']}-{row['end_frame']}")
    ax.set_xlabel("frame")
    ax.set_ylabel("knee angle (deg)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in f"{row['skater']}_{row['phase']}_{row['start_frame']}")
    fig.savefig(os.path.join(PLOT_DIR, f"{safe}.png"), dpi=90)
    plt.close(fig)


def main():
    results, skipped = [], []
    for row in load_labeled_segments():
        if row["phase"] not in MOVING_PHASES:
            continue
        res, why = analyze_segment(row)
        if res is not None and why is None:
            results.append(res)
        else:
            skipped.append((row["skater"], row["phase"], f"{row['start_frame']}-{row['end_frame']}", why))

    pd.set_option("display.width", 200)
    df = pd.DataFrame(results)

    print(f"\n{'='*78}")
    print("PHASE 5d: STRIDE RHYTHM PER SEGMENT")
    print(f"{'='*78}")
    if df.empty:
        print("No segment had enough strides to measure rhythm.")
    else:
        cols = ["skater", "phase", "segment", "fps", "duration_s", "n_intervals",
                "stride_period_s", "cadence_per_min", "rhythm_cv", "lr_phase_offset",
                "cv_sensitivity", "stable"]
        print(df[cols].round(3).to_string(index=False))
        df.to_csv(OUTPUT_PATH, index=False)

        stable = df[df["stable"]]
        print(f"\ncv_sensitivity = rhythm_cv at trough-depth settings {SENSITIVITY_REL}.")
        print(f"STABLE = moves <= {STABLE_CV_RANGE} across them. Only STABLE rows are trustworthy.")
        print(f"\nPer skater x phase, STABLE segments only ({len(stable)}/{len(df)}):")
        if stable.empty:
            print("  none")
        else:
            agg = stable.groupby(["phase", "skater"])[["stride_period_s", "cadence_per_min",
                                                       "rhythm_cv", "lr_phase_offset"]].mean()
            print(agg.round(3).to_string())

    print(f"\nSkipped ({len(skipped)} segments too short or unusable):")
    for s in skipped:
        print(f"  {s[0]:18s} {s[1]:18s} {s[2]:10s} -- {s[3]}")

    print(f"\nPlots of every detection -> {PLOT_DIR}/  (CHECK these by eye before trusting)")
    print("READING THIS:")
    print("  - rhythm_cv: 0.05 = very even strides, >0.25 = irregular (or bad detection).")
    print("  - lr_phase_offset near 0.5 = legs alternate evenly.")
    print("  - A stride period far outside ~0.6-2.0 s suggests slow-motion footage or")
    print("    missed/extra peaks -- check the plot for that segment.")
    if not df.empty:
        print(f"\nResults saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
