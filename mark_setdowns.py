"""
Phase 8b (10/29): player for marking stroke set-downs by hand.

Plays each clip in stroke_rate/clips/ and records the video frame number
whenever M is pressed, straight into stroke_rate/hand_counts.csv
(setdown_frames column). No automatic counts are shown.

Keys:
    SPACE      pause / play
    + / -      faster / slower (starts at half speed)
    D / A      one frame forward / back (pauses)
    M          mark a set-down at the current frame
    U          undo the last mark
    C          "can't tell" for this clip, go to the next
    N          save this clip's marks, go to the next
    R          restart this clip (keeps marks)
    Q          save and quit

Usage:
    python -m mark_setdowns                 # every clip without marks yet
    python -m mark_setdowns --clip test_05  # one clip (redo)
    python -m mark_setdowns --set test      # only the test clips
"""

import os
import argparse
import cv2
import pandas as pd

COUNTS = "stroke_rate/hand_counts.csv"
CLIPS = "stroke_rate/clips"
VIDEO_FPS = 30.0
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]


def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def mark_clip(row, frames):
    """Returns (marks, status) with status 'next', 'cant', or 'quit'."""
    marks = [int(x) for x in str(row.setdown_frames).split()] if pd.notna(row.setdown_frames) else []
    i, paused, sp = 0, True, SPEEDS.index(0.5)
    win = "mark set-downs"
    while True:
        f = frames[i].copy()
        vf = int(row.start_frame) + i
        h = f.shape[0]
        info = (f"{row['clip']}  {'PAUSED' if paused else 'playing'}  speed {SPEEDS[sp]}x  "
                f"marks: {len(marks)}  [M mark  U undo  SPACE play  A/D step  +/- speed  N next  C can't tell  Q quit]")
        cv2.rectangle(f, (0, h - 40), (f.shape[1], h), (0, 0, 0), -1)
        cv2.putText(f, info, (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if vf in marks:
            cv2.putText(f, "MARKED", (f.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        # 10/29: first pass had a mark on the last frame of 10 of 15 clips (M pressed after the
        # clip stopped), so the end is shown plainly
        if i == len(frames) - 1:
            cv2.putText(f, "END OF CLIP: N = next, R = replay", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow(win, f)
        k = cv2.waitKey(0 if paused else max(1, int(1000 / (VIDEO_FPS * SPEEDS[sp])))) & 0xFF
        if k == ord(" "):
            paused = not paused
        elif k in (ord("+"), ord("=")):
            sp = min(sp + 1, len(SPEEDS) - 1)
        elif k in (ord("-"), ord("_")):
            sp = max(sp - 1, 0)
        elif k in (ord("d"), ord("D")):
            paused, i = True, min(i + 1, len(frames) - 1)
        elif k in (ord("a"), ord("A")):
            paused, i = True, max(i - 1, 0)
        elif k in (ord("m"), ord("M")):
            if vf not in marks:
                marks = sorted(marks + [vf])
        elif k in (ord("u"), ord("U")):
            if marks:
                marks.pop()
        elif k in (ord("r"), ord("R")):
            i = 0
        elif k in (ord("n"), ord("N")):
            return marks, "next"
        elif k in (ord("c"), ord("C")):
            return [], "cant"
        elif k in (ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            return marks, "quit"
        elif not paused:
            if i < len(frames) - 1:
                i += 1
            else:
                paused = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--set", choices=["tune", "test"])
    args = ap.parse_args()
    d = pd.read_csv(COUNTS, dtype=str)
    if args.clip:
        todo = d.index[d["clip"] == args.clip]
    else:
        todo = d.index[d["setdown_frames"].isna() & ~d["note"].fillna("").str.contains("can't tell")]
        if args.set:
            todo = [j for j in todo if d.at[j, "set"] == args.set]
    for j in todo:
        row = d.loc[j]
        frames = load_frames(os.path.join(CLIPS, row["clip"] + ".mp4"))
        marks, status = mark_clip(row, frames)
        if status == "cant":
            d.at[j, "setdown_frames"] = None
            d.at[j, "note"] = "can't tell"
        elif marks:
            d.at[j, "setdown_frames"] = " ".join(map(str, marks))
        d.to_csv(COUNTS, index=False)
        print(row["clip"], status, " ".join(map(str, marks)))
        if status == "quit":
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
