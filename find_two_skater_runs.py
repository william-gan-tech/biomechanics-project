"""
Phase 6c (10/11): find stretches where BOTH skaters are in frame -- the
footage Phase 5 had to skip -- from the existing race scans.

scan_full_race.py logged, for every sampled frame, how many people MediaPipe
detected (n_candidates). Phase 5 kept only n_candidates == 1. Here we look for
runs where n_candidates == 2 for at least MIN_SECONDS (allowing brief
dropouts), within a given frame window (one pair's race).

These are CANDIDATES for the dual tracker; pure pieces then decides what is
safe to keep, and the researcher reviews the pieces.

Usage:
    python -m find_two_skater_runs --scan race_scan_inzell_2026_5000m.csv --fps 25 --start 0 --end 9300 --name inzell
"""

import argparse
import pandas as pd

MIN_SECONDS = 2.5
MAX_GAP_SAMPLES = 2


def runs(scan, fps, start, end):
    d = scan[(scan.frame >= start) & (scan.frame <= end)].sort_values("frame")
    good = (d.n_candidates == 2).values
    frames = d.frame.values
    out, s, last, bad = [], None, None, 0
    for i, ok in enumerate(good):
        if ok:
            s = i if s is None else s
            last, bad = i, 0
        elif s is not None:
            bad += 1
            if bad > MAX_GAP_SAMPLES:
                out.append((s, last))
                s, bad = None, 0
    if s is not None:
        out.append((s, last))
    rows = []
    for a, b in out:
        dur = (frames[b] - frames[a]) / fps
        if dur >= MIN_SECONDS:
            rows.append({"start_frame": int(frames[a]), "end_frame": int(frames[b]),
                         "duration_s": round(dur, 1), "clock_min": round(frames[a] / fps / 60, 2)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10 ** 9)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    r = runs(pd.read_csv(a.scan), a.fps, a.start, a.end)
    r.to_csv(f"dual_tracking/{a.name}_two_skater_runs.csv", index=False)
    print(f"{len(r)} two-skater runs >= {MIN_SECONDS}s, total {r.duration_s.sum() if len(r) else 0:.0f} s")
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
