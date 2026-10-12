"""
Phase 6c (10/11): review sheets for pure pieces recovered by the dual tracker.

Takes a dual_track.py tracks CSV, splits it into pure pieces
(score_dual_tracking.pure_pieces -- each piece is one person; confirmed 100%
on 9 checked clips), keeps pieces of at least MIN_SECONDS, and draws ONE ROW
per piece: every Nth frame with only that piece's box, labelled P<number>.

The researcher then says, per piece: which skater (or 'not a skater'),
phase (corner/straightaway) and camera shot type -- as in Phase 5, but with
the box already marking who is being followed.

Output:
  dual_tracking/<name>_pieces.csv         piece, track_id, first/last frame, seconds
  dual_tracking/<name>_pieces_<k>.jpg      sheets of up to ROWS_PER_SHEET pieces

Usage:
    python -m make_piece_sheets --video data/inzell_2026_5000m.mp4 --name inzell_full_race --fps 25
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd

from score_dual_tracking import pure_pieces, load_cuts

MIN_SECONDS = 1.5
MIN_HEIGHT_FRAC = 0.15   # median box height vs frame height: smaller = too small to measure knees (wide shots)
# Note (10/11): people in the stands (shown in crowd shots) are NOT filtered
# automatically -- the researcher marks them "not a skater" on the sheets.
# 10/11: fewer, larger tiles per sheet -- the first version (10 pieces x 8 small
# frames per sheet) was too dense for the researcher to read clearly
TILES_PER_ROW = 5
ROWS_PER_SHEET = 4
TILE_W = 380


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--fps", type=float, required=True)
    a = p.parse_args()

    tracks = pd.read_csv(f"dual_tracking/{a.name}_tracks.csv")
    cuts = load_cuts(a.name)
    if not cuts:
        print("WARNING: no camera cuts found for this clip -- run detect_cuts.py first, or pieces may cross cuts")
    pieces = pure_pieces(tracks, cuts)
    rows = [{"frame": f, "track_id": t, "piece": pid} for (f, t), pid in pieces.items()]
    df = pd.DataFrame(rows).merge(tracks, on=["frame", "track_id"])
    summary = df.groupby("piece").agg(track_id=("track_id", "first"), first_frame=("frame", "min"),
                                      last_frame=("frame", "max")).reset_index()
    summary["seconds"] = ((summary.last_frame - summary.first_frame + 1) / a.fps).round(1)
    cap0 = cv2.VideoCapture(a.video)
    frame_h = cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap0.release()
    df["h"] = df.y2 - df.y1
    df["cx"] = (df.x1 + df.x2) / 2
    stats = df.groupby("piece").agg(med_h=("h", "median"), med_w=("x2", "median"),
                                    cx_range=("cx", lambda s: s.max() - s.min()))
    stats["med_w"] = df.assign(w=df.x2 - df.x1).groupby("piece").w.median()
    summary = summary.merge(stats, left_on="piece", right_index=True)
    n_before = len(summary[summary.seconds >= MIN_SECONDS])
    keep = ((summary.seconds >= MIN_SECONDS) & (summary.med_h >= MIN_HEIGHT_FRAC * frame_h))
    summary = summary[keep].sort_values("first_frame").reset_index(drop=True)
    print(f"size filter: {n_before} -> {len(summary)} pieces (dropped boxes under {MIN_HEIGHT_FRAC:.0%} of frame height)")
    summary["piece_no"] = range(1, len(summary) + 1)
    summary["video_time"] = [f"{int(f / a.fps // 60)}:{f / a.fps % 60:04.1f}" for f in summary.first_frame]
    summary.to_csv(f"dual_tracking/{a.name}_pieces.csv", index=False)
    print(f"{len(summary)} pieces >= {MIN_SECONDS}s (of {df.piece.nunique()} pure pieces), "
          f"total {summary.seconds.sum():.0f} s")

    # frames to draw, per piece: TILES_PER_ROW evenly spaced frames
    want = {}
    for r in summary.itertuples():
        pf = df[df.piece == r.piece].sort_values("frame")
        idx = np.linspace(0, len(pf) - 1, min(TILES_PER_ROW, len(pf))).astype(int)
        for row in pf.iloc[idx].itertuples():
            want.setdefault(int(row.frame), []).append((r.piece_no, row.x1, row.y1, row.x2, row.y2))

    tiles = {r.piece_no: [] for r in summary.itertuples()}
    cap = cv2.VideoCapture(a.video)
    start = min(want) if want else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n, last = start, max(want) if want else -1
    while n <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for pno, x1, y1, x2, y2 in want.get(n, []):
            img = frame.copy()
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 5)
            t = cv2.resize(img, (TILE_W, int(img.shape[0] * TILE_W / img.shape[1])))
            cv2.rectangle(t, (0, 0), (TILE_W, 32), (0, 0, 0), -1)
            cv2.putText(t, f"P{pno}   frame {n}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            tiles[pno].append(t)
        n += 1
    cap.release()

    blank = None
    sheet_rows, k = [], 0
    for pno in sorted(tiles):
        row = tiles[pno]
        if not row:
            continue
        blank = np.zeros_like(row[0]) if blank is None else blank
        while len(row) < TILES_PER_ROW:
            row.append(blank)
        sheet_rows.append(np.hstack(row))
        if len(sheet_rows) == ROWS_PER_SHEET:
            cv2.imwrite(f"dual_tracking/{a.name}_pieces_{k:02d}.jpg", np.vstack(sheet_rows))
            sheet_rows, k = [], k + 1
    if sheet_rows:
        cv2.imwrite(f"dual_tracking/{a.name}_pieces_{k:02d}.jpg", np.vstack(sheet_rows))
        k += 1
    print(f"{k} review sheet(s) -> dual_tracking/{a.name}_pieces_*.jpg")


if __name__ == "__main__":
    main()
