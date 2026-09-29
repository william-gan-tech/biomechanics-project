"""
Whole-race scan: finds long, clean, single-skater stretches in a full race
video, so labeling for Phase 5f (fatigue, early vs late race) only needs
visual checks on a short list of candidates instead of browsing thousands
of frames.

Reads the video ONCE, sequentially (MediaPipe VIDEO mode needs temporal
continuity -- see the 9/24 fix in check_same_skater_identity.py). Per sampled
frame it records:
  - n_candidates        number of detected people (want exactly 1)
  - full_body           both ankles visible and inside the frame (sit height,
                        knee angle and stride detection all need the legs)
  - body_height_px      detected person's height in pixels; very large =
                        close-up (the 9/28 calibration bug came from a
                        close-up), very small = too far away to trust

A "clean run" is a stretch where every sample has exactly one full-body
person of a usable size, allowing brief dropouts. Runs shorter than
--min-seconds are dropped: stride rhythm needs 4+ strides (9/28 finding).

Output:
  race_scan_<name>.csv          per-sample log
  race_scan_<name>_runs.csv     clean runs with frame ranges and timestamps
  race_scan_<name>/             one thumbnail at the middle of each run --
                                open these to label corner vs straightaway
                                and read the lap counter

KNOWN LIMITATION (9/28): runs can contain CAMERA CUTS (e.g. side rail-cam
switching to a front close-up). All 5 first-pass Inzell candidates did.
Automatic cut detection was attempted and not adopted: whole-frame color
correlation stayed ~0.99 across real cuts between similar-looking arena
shots, and skater size/position jumps of 1.3-1.5x also occur within single
shots from detection jitter. `frame_corr` and `center_x` are still logged
per sample for future work. Until then, check every run for cuts by eye
(dense frame sheet) before logging it.

This is a CANDIDATE FINDER only. Every run still goes through the normal
verification (identity check, visual phase check) before being logged.

Usage:
    python -m scan_full_race --video data/inzell_2026_5000m.mp4
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from pipeline_engine import _resolve_pose_model_path

MIN_BODY_FRAC = 0.12     # person must be >= 12% of frame height (too far = unreliable)
MAX_BODY_FRAC = 0.85     # > 85% of frame height = close-up, legs likely cut off
ANKLE_VIS = 0.5
MAX_DROPOUT_SAMPLES = 2  # tolerate this many consecutive bad samples inside a run


def scan(video_path, sample_stride, start_frame=0, end_frame=None):
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_resolve_pose_model_path()),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frame_num = 0
    prev_hist = None
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret or (end_frame is not None and frame_num > end_frame):
                break
            # Frames before start_frame are decoded but not analyzed: seeking is
            # unreliable in this MP4 (yt-dlp MPEG-TS warning), sequential reads aren't
            if frame_num >= start_frame and frame_num % sample_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # Timestamp from frame number: robust to the malformed MP4
                # timestamps yt-dlp warned about, and always increasing
                result = landmarker.detect_for_video(image, int(frame_num * 1000 / fps))
                cands = [p for p in result.pose_landmarks if len(p) > 28] if result.pose_landmarks else []

                # Whole-frame color similarity to the previous sample, for
                # camera-cut detection (a cut replaces the entire scene at once)
                small = cv2.resize(frame, (160, 90))
                hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None,
                                    [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                frame_corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) if prev_hist is not None else 1.0
                prev_hist = hist

                row = {"frame": frame_num, "time_s": frame_num / fps, "n_candidates": len(cands),
                       "full_body": False, "body_height_px": np.nan, "center_x": np.nan,
                       "frame_corr": frame_corr}
                if len(cands) >= 1:
                    lm = cands[0]
                    ys = [p.y for p in lm]
                    row["body_height_px"] = (max(ys) - min(ys)) * height
                    row["center_x"] = float(np.mean([p.x for p in lm]))
                    ankles_ok = all(lm[i].visibility > ANKLE_VIS and 0.0 <= lm[i].y <= 1.0 for i in (27, 28))
                    row["full_body"] = bool(ankles_ok)
                rows.append(row)
            frame_num += 1
            if frame_num % 1000 == 0:
                print(f"  scanned {frame_num}/{total} frames...")
    cap.release()
    return pd.DataFrame(rows), fps, height


def find_runs(df, fps, height, min_seconds):
    good = ((df["n_candidates"] == 1) & df["full_body"]
            & (df["body_height_px"] >= MIN_BODY_FRAC * height)
            & (df["body_height_px"] <= MAX_BODY_FRAC * height)).values
    frames = df["frame"].values

    runs, start, last_good, bad_streak = [], None, None, 0
    for i, ok in enumerate(good):
        if ok:
            if start is None:
                start = i
            last_good, bad_streak = i, 0
        elif start is not None:
            bad_streak += 1
            if bad_streak > MAX_DROPOUT_SAMPLES:
                runs.append((start, last_good))
                start, bad_streak = None, 0
    if start is not None:
        runs.append((start, last_good))

    out = []
    for a, b in runs:
        dur = (frames[b] - frames[a]) / fps
        if dur < min_seconds:
            continue
        seg = df.iloc[a:b + 1]
        out.append({
            "start_frame": int(frames[a]), "end_frame": int(frames[b]),
            "duration_s": round(dur, 2),
            "start_time": f"{int(frames[a] / fps // 60)}:{frames[a] / fps % 60:05.2f}",
            "clean_fraction": round(float(good[a:b + 1].mean()), 3),
            "median_body_px": round(float(seg["body_height_px"].median()), 1),
        })
    return pd.DataFrame(out)


def save_thumbnails(video_path, runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    for _, r in runs.iterrows():
        mid = (r["start_frame"] + r["end_frame"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret:
            continue
        label = f"frames {r['start_frame']}-{r['end_frame']} ({r['duration_s']}s) @ {r['start_time']}"
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        small = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
        cv2.imwrite(os.path.join(out_dir, f"run_{r['start_frame']:06d}.jpg"), small)
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--stride", type=int, default=2, help="analyze every Nth frame")
    parser.add_argument("--min-seconds", type=float, default=2.5)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for output names, e.g. 'early'")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0] + (f"_{args.tag}" if args.tag else "")
    print(f"Scanning {args.video} sequentially (every {args.stride} frames, "
          f"frames {args.start_frame}-{args.end_frame or 'end'})...")
    df, fps, height = scan(args.video, args.stride, args.start_frame, args.end_frame)
    df.to_csv(f"race_scan_{name}.csv", index=False)

    runs = find_runs(df, fps, height, args.min_seconds)
    runs_path = f"race_scan_{name}_runs.csv"
    runs.to_csv(runs_path, index=False)

    print(f"\nSampled {len(df)} frames. Candidate-count breakdown:")
    print(df["n_candidates"].value_counts().sort_index().to_string())
    print(f"Full-body single-person samples: {((df['n_candidates'] == 1) & df['full_body']).mean():.1%}")

    if runs.empty:
        print(f"\nNo clean runs >= {args.min_seconds}s found.")
        return
    print(f"\n{len(runs)} clean single-skater runs >= {args.min_seconds}s:")
    print(runs.to_string(index=False))
    save_thumbnails(args.video, runs, f"race_scan_{name}")
    print(f"\nSaved -> {runs_path} and thumbnails in race_scan_{name}/")
    print("NEXT: open each thumbnail, note corner/straightaway + lap number, then run the")
    print("usual identity check on the runs you want before logging them.")


if __name__ == "__main__":
    main()
