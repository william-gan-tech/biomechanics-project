"""
Phase 6c (10/13): turn the researcher's labelled pure pieces into segments
that the Phase 5 analyses can use (same columns as labeled_segments.csv).

Each piece labelled with a skater AND a phase (corner/straightaway) becomes
one segment. Pose comes from piece_pose.py (measured on the tracker box's
crop), so each row carries `pose_source` and `piece_no`;
compare_to_elite_reference.get_segment_frames reads those instead of the
whole-frame Phase 5 features.

Lap numbers use the Beijing pairs_map rule (on-screen race clock):
    lap = floor((clock_seconds - 18) / 30) + 1
with clock_seconds = (frame - race_start_frame) / fps. Race start frames
were worked back from the clock times in pairs_map.md. Pieces before the
start or after 12.5 laps are dropped.

Recovered segments are kept in their OWN file (dual_tracking/
recovered_segments.csv), not labeled_segments.csv, so every Phase 5 result
can still be reproduced without them; 5f adds them only when asked
(PHASE5F_RECOVERED=1).

Usage:
    python -m recovered_segments
"""

import os
import math
import pandas as pd

FPS = 30.0
VIDEO = "data/beijing_2022_5000m.mp4"
# race clock 0:00 frame, from pairs_map.md (e.g. pair 4: 53742 at 1:24 -> 51222)
RACES = {
    "beijing_p3_engebraaten_trofimov": 37083,
    "beijing_p4_wenger_cepuran": 51222,
    "beijing_p10_vanderpoel_swings": 180216,
}
FULL_NAMES = {
    "Wenger": "Livio Wenger", "Cepuran": "Ethan Cepuran",
    "Engebraaten": "Hallgeir Engebraaten", "Trofimov": "Sergey Trofimov",
    "van der Poel": "Nils van der Poel", "Swings": "Bart Swings",
}
PHASES = {"corner": "corner", "straight": "straightaway", "straightaway": "straightaway"}
OUTPUT = "dual_tracking/recovered_segments.csv"


def lap_of(frame, race_start):
    clock = (frame - race_start) / FPS
    if clock < 0:
        return None
    lap = max(1, math.floor((clock - 18) / 30) + 1)
    return lap if lap <= 12 else None


def main():
    rows = []
    for name, start in RACES.items():
        path = f"dual_tracking/{name}_piece_labels.csv"
        if not os.path.exists(path):
            continue
        lab = pd.read_csv(path, dtype=str).fillna("")
        # a second usable part of a piece (e.g. front corner, then side corner):
        # extra rows in <name>_piece_extra.csv, same columns as the labels file
        extra = f"dual_tracking/{name}_piece_extra.csv"
        if os.path.exists(extra):
            lab = pd.concat([lab, pd.read_csv(extra, dtype=str).fillna("")], ignore_index=True)
        for r in lab.itertuples():
            skater, phase = FULL_NAMES.get(r.skater), PHASES.get(r.phase.strip().lower())
            if not skater or not phase:
                continue   # not a skater, or phase not labelled yet
            first, last = int(r.first_frame), int(r.last_frame)
            # part of a piece only, where the phase or shot changes within it
            # (use_from/use_to, set from the researcher's notes)
            if getattr(r, "use_from", ""):
                first = int(r.use_from)
            if getattr(r, "use_to", ""):
                last = int(r.use_to)
            lap = lap_of((first + last) // 2, start)
            if lap is None:
                continue
            rows.append({
                "skater": skater, "video_path": VIDEO, "phase": phase,
                "start_frame": first, "end_frame": last, "date_confirmed": "",
                "notes": f"lap {lap}, recovered piece P{r.piece_no} ({name}); researcher-labelled",
                "shot": r.shot, "pose_source": f"dual_tracking/{name}_piece_pose.csv",
                "piece_no": int(r.piece_no), "lap": lap,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    if out.empty:
        print("no labelled pieces with skater + phase yet")
        return
    print(f"{len(out)} recovered segments -> {OUTPUT}")
    print(out.groupby(["skater", "phase"]).lap.agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
