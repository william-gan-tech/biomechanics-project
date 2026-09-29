"""
Frame sheet: a grid of every Nth frame from a range, labeled with frame
numbers, for checking a segment by eye before logging it -- camera cuts,
phase transitions (straightaway -> corner), a second skater, occlusions.

Added 9/28 after all 5 first-pass Inzell race-scan candidates turned out to
contain a camera cut or a phase transition that the automated scan missed.

Reads frames sequentially rather than seeking (seeking is unreliable in
yt-dlp downloads, which can have malformed MP4 timestamps).

Usage:
    python -m make_frame_sheet --video data/inzell_2026_5000m.mp4 --start 1632 --end 1700
    python -m make_frame_sheet --video data/inzell_2026_5000m.mp4 --start 1632 --end 1700 --step 2 --open
"""

import os
import argparse
import cv2
import numpy as np


def make_sheet(video_path, start, end, step=4, per_row=8, tile_width=240):
    cap = cv2.VideoCapture(video_path)
    tiles, n = [], 0
    while n <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if n >= start and (n - start) % step == 0:
            h = int(frame.shape[0] * tile_width / frame.shape[1])
            tile = cv2.resize(frame, (tile_width, h))
            cv2.putText(tile, str(n), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(tile)
        n += 1
    cap.release()
    if not tiles:
        return None
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step", type=int, default=4, help="show every Nth frame")
    parser.add_argument("--out-dir", default="frame_sheets")
    parser.add_argument("--open", action="store_true", help="open the image when done")
    args = parser.parse_args()

    sheet = make_sheet(args.video, args.start, args.end, args.step)
    if sheet is None:
        print(f"No frames read in {args.start}-{args.end} -- check the range and video path.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0]
    path = os.path.join(args.out_dir, f"{name}_{args.start}-{args.end}.jpg")
    cv2.imwrite(path, sheet)
    print(f"Saved -> {path}")
    print("Check: same skater throughout? any camera cut? phase changes (straight -> corner)?")
    print("       anyone blocking the skater? If so, trim the range and re-run.")
    if args.open:
        os.startfile(os.path.abspath(path))


if __name__ == "__main__":
    main()
