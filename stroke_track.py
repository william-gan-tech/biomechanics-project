"""
Phase 8c (10/29): pose tracking for the stroke-count clips.

Runs the Phase 7 tracker (doublepush_measures.track: MediaPipe pose, restarted
at camera cuts, heel/toe foot points) on each clip in stroke_rate/clips/ and
saves the per-frame values with the original video frame numbers
(clip frame + start_frame), so they line up with the hand-marked set-downs.

The clips are used instead of the full race video because seeking in the
2 h broadcast is unreliable (yt-dlp download); each clip holds every frame of
its segment, only slowed.

Output: stroke_rate/frames_<clip>.csv

Usage:
    python -m stroke_track
"""

import os
import pandas as pd

from doublepush_measures import track

COUNTS = "stroke_rate/hand_counts.csv"
CLIPS = "stroke_rate/clips"
VIDEO_FPS = 30.0


def main():
    for _, s in pd.read_csv(COUNTS).iterrows():
        out = f"stroke_rate/frames_{s['clip']}.csv"
        if os.path.exists(out):
            continue
        d, _ = track(os.path.join(CLIPS, s["clip"] + ".mp4"), 0, int(s.end_frame - s.start_frame))
        d["frame"] += int(s.start_frame)
        d.to_csv(out, index=False)
        print(s["clip"], len(d), "of", int(s.end_frame - s.start_frame) + 1, "frames with pose",
              f"leg {d.leg_px.median():.0f} px" if len(d) else "")


if __name__ == "__main__":
    main()
