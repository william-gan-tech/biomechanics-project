"""
Camera shot-type labelling for the Phase 5f segments (9/30).

The 9/30 viewpoint check found that late-race segments are filmed from a
more frontal angle than early ones, and that this explains most of the
"more upright late in the race" pattern. Its viewpoint measure (projected hip
width) partly depends on the skater's own body, so it may over-correct. A
shot-type label made by eye does not depend on the skater's body at all.

This script makes one sheet per video with the MIDDLE frame of every
lap-labeled 5f segment, numbered, and a CSV template to fill in:

  shot_labels.csv   columns: id, skater, video_path, phase, start_frame,
                    end_frame, lap, shot_type, labeler

shot_type values:
  side   camera roughly perpendicular to the skater (profile view)
  front  skater coming toward or going away from the camera
  wide   high/wide shot, skater small in frame

Existing labels in shot_labels.csv are kept; only new segments are added.

Usage:
    python -m make_shot_label_sheets
"""

import os
import csv
import cv2
import numpy as np
import pandas as pd

from validate_fatigue_form_degradation import lap_segments

LABELS_PATH = "shot_labels.csv"
OUT_DIR = os.path.join("frame_sheets", "shot_labels")
FIELDS = ["id", "skater", "video_path", "phase", "start_frame", "end_frame", "lap", "shot_type", "labeler"]


def main():
    segs = lap_segments().reset_index(drop=True)
    segs["start_frame"] = segs["start_frame"].astype(int)
    segs["end_frame"] = segs["end_frame"].astype(int)
    segs = segs.sort_values(["video_path", "start_frame"]).reset_index(drop=True)
    segs["id"] = range(1, len(segs) + 1)

    existing = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                existing[(r["video_path"], r["start_frame"], r["end_frame"])] = r

    rows = []
    for _, s in segs.iterrows():
        key = (s.video_path, str(s.start_frame), str(s.end_frame))
        old = existing.get(key, {})
        rows.append({"id": s.id, "skater": s.skater, "video_path": s.video_path, "phase": s.phase,
                     "start_frame": s.start_frame, "end_frame": s.end_frame, "lap": s.lap,
                     "shot_type": old.get("shot_type", ""), "labeler": old.get("labeler", "")})
    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    for video, g in segs.groupby("video_path"):
        mids = {int((a + b) // 2): i for i, a, b in zip(g.id, g.start_frame, g.end_frame)}
        tiles = {}
        cap = cv2.VideoCapture(video)
        n, last = 0, max(mids)
        while n <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if n in mids:
                t = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
                sid = mids[n]
                row = g[g.id == sid].iloc[0]
                label = f"#{sid} {row.skater.split()[-1]} L{row.lap}"
                cv2.rectangle(t, (0, 0), (320, 26), (0, 0, 0), -1)
                cv2.putText(t, label, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                tiles[sid] = t
            n += 1
        cap.release()
        ordered = [tiles[i] for i in sorted(tiles)]
        while len(ordered) % 6:
            ordered.append(np.zeros_like(ordered[0]))
        sheet = np.vstack([np.hstack(ordered[i:i + 6]) for i in range(0, len(ordered), 6)])
        name = os.path.splitext(os.path.basename(video))[0]
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_shots.jpg"), sheet)
        print(f"Saved -> {OUT_DIR}/{name}_shots.jpg ({len(tiles)} segments)")
    print(f"Template -> {LABELS_PATH} ({len(rows)} segments; fill in shot_type as side / front / wide)")


if __name__ == "__main__":
    main()
