"""
Phase 6a/6b: multi-object tracking of both skaters (10/8).

Runs a person detector (YOLO) with a multi-object tracker (ByteTrack, via
the `ultralytics` package) over a frame range, so every person gets a
persistent track ID. This replaces the 9/22 approach (MediaPipe num_poses=2 +
hand-built matching), which failed on ~21% duplicate detections and identity
swaps between similar-looking skaters.

Outputs, per clip:
  dual_tracking/<name>_tracks.csv   frame, track_id, box, confidence
  dual_tracking/<name>_review.jpg   every Nth frame with each track's box and
                                    ID drawn -- for the researcher's identity
                                    check (ground truth for 6a)

The review sheet is how accuracy is measured: the researcher says which
track ID is which skater, and lists frames where an ID jumps to the other
skater (identity swap) or a skater has no box (lost). Target (fixed 10/8,
before testing): identity correct on >= 95% of checked frames.

Frames are reached by seeking to the clip start and then reading
sequentially; frame numbers on the sheet and in the CSV come from the same
read, so they are internally consistent.

Usage:
    python -m dual_track --video data/beijing_2022_5000m.mp4 --start 126270 --end 126489 --name bergsma_ichinohe
    python -m dual_track --batch dual_tracking/test_clips.csv
"""

import os
import csv
import argparse
import cv2
import numpy as np
import pandas as pd

OUT_DIR = "dual_tracking"
MODEL = "yolo11n.pt"          # smallest YOLO model; downloaded on first use (~6 MB)
REVIEW_STEP = 6
COLORS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 255, 255)]


def track_clip(video, start, end, name, model_name=MODEL, min_box_frac=0.08):
    from ultralytics import YOLO
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows, review = [], []
    n = start
    while n <= end:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)[0]
        boxes = []
        if res.boxes is not None and res.boxes.id is not None:
            for (x1, y1, x2, y2), tid, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy(),
                                                   res.boxes.conf.cpu().numpy()):
                # skip tiny boxes (distant crowd / officials) -- skaters of interest are larger
                if (y2 - y1) < min_box_frac * h_img:
                    continue
                boxes.append((int(tid), float(x1), float(y1), float(x2), float(y2), float(conf)))
                rows.append({"frame": n, "track_id": int(tid), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})
        if (n - start) % REVIEW_STEP == 0:
            img = frame.copy()
            for tid, x1, y1, x2, y2, conf in boxes:
                c = COLORS[tid % len(COLORS)]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 3)
                cv2.putText(img, str(tid), (int(x1), max(30, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, c, 4)
            tile = cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])))
            cv2.rectangle(tile, (0, 0), (110, 24), (0, 0, 0), -1)
            cv2.putText(tile, str(n), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            review.append(tile)
        n += 1
    cap.release()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"{name}_tracks.csv"), index=False)
    if review:
        while len(review) % 6:
            review.append(np.zeros_like(review[0]))
        sheet = np.vstack([np.hstack(review[i:i + 6]) for i in range(0, len(review), 6)])
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_review.jpg"), sheet)

    if df.empty:
        return {"name": name, "frames": n - start, "tracks": 0}
    counts = df.groupby("track_id").frame.nunique().sort_values(ascending=False)
    return {"name": name, "frames": n - start, "tracks": len(counts),
            "top_tracks": ", ".join(f"ID {i}: {c} frames" for i, c in counts.head(5).items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--name")
    p.add_argument("--batch", help="CSV with name,video,start,end")
    a = p.parse_args()
    jobs = []
    if a.batch:
        with open(a.batch, encoding="utf-8-sig") as f:
            jobs = [(r["video"], int(r["start"]), int(r["end"]), r["name"]) for r in csv.DictReader(f)]
    else:
        jobs = [(a.video, a.start, a.end, a.name)]
    for v, s, e, n in jobs:
        r = track_clip(v, s, e, n)
        print(f"{r['name']:28s} frames {r['frames']:4d}  tracks {r['tracks']:3d}  | {r.get('top_tracks', '')}", flush=True)


if __name__ == "__main__":
    main()
