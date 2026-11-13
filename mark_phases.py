"""
Phase 9a (11/12): player for marking race phases across a whole race.

Plays one pair's race window straight through and records, for one named
skater, the frame of each start, corner entry, corner exit and finish, into
phase9/phase_marks.csv. These frame-precise boundaries are the labels the
phase model is trained and tested on (every frame between an entry and the
next exit is corner; between an exit and the next entry, straight).

A whole race is too long to hold in memory, so frames are read in order and
the last BUFFER frames are kept for stepping back (seeking is unreliable in
yt-dlp downloads). The race clock and the skater's lap are shown from the
Phase 8 clock offsets and the lap times in stroke_rate/lap_times.csv and
pacing/lap_times.csv.

Keys:
    SPACE      pause / play
    + / -      faster / slower
    D / A      one frame forward / back (pauses)
    L / J      one second forward / back (pauses)
    s / f      start / finish
    e / x      corner entry / corner exit, seen on screen
    E / X      corner entry / corner exit, not seen (estimated, e.g. skater off camera)
    U          undo the last mark pressed (this session; else the latest frame)
    Q          save and quit (marks are also saved on every key)

Usage:
    python -m mark_phases --pair pair10_vanderpoel --skater "Nils van der Poel"
    python -m mark_phases --pair pair10_vanderpoel --skater "Bart Swings"   # the opponent
Restarts 5 s before the skater's last mark unless --from is given.
"""

import os
import argparse
from collections import deque
from datetime import datetime
import cv2
import pandas as pd

from lap_graphics import VIDEO, WINDOWS
from stroke_rate_laps import CLOCK_OFFSET

MARKS = "phase9/phase_marks.csv"
LAP_FILES = ["stroke_rate/lap_times.csv", "pacing/lap_times.csv"]
FPS = 30.0
BUFFER = 300
SHOW = (960, 540)
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
EVENTS = {ord("s"): ("start", 1), ord("f"): ("finish", 1), ord("e"): ("entry", 1), ord("x"): ("exit", 1),
          ord("E"): ("entry", 0), ord("X"): ("exit", 0)}
COLUMNS = ["race", "pair", "skater", "frame", "event", "seen", "clock_s", "lap", "marked"]


def lap_finder(skater):
    for p in LAP_FILES:
        d = pd.read_csv(p)
        cum = d[d.skater == skater].sort_values("lap").cum_s.dropna().tolist()
        if cum:
            return lambda clock: next((i for i, c in enumerate(cum, 1) if clock < c), None)
    return lambda clock: None


def expected(marks):
    """What should come next, from the skater's marks so far."""
    if not marks:
        return "start"
    last = marks[-1]["event"]
    return {"start": "entry", "exit": "entry or finish", "entry": "exit", "finish": "nothing (done)"}[last]


def load(pair, skater):
    if os.path.exists(MARKS):
        d = pd.read_csv(MARKS)
    else:
        d = pd.DataFrame(columns=COLUMNS)
    mine = d[(d.pair == pair) & (d.skater == skater)].sort_values("frame").to_dict("records")
    return d[~((d.pair == pair) & (d.skater == skater))], mine


def save(others, mine):
    os.makedirs(os.path.dirname(MARKS), exist_ok=True)
    pd.concat([others, pd.DataFrame(mine, columns=COLUMNS)]).to_csv(MARKS, index=False)


def run(pair, skater, start=None, video=VIDEO):
    a, b = WINDOWS[pair]
    offset = CLOCK_OFFSET[pair]
    lap_of = lap_finder(skater)
    others, mine = load(pair, skater)
    if start is None:
        start = max(a, int(mine[-1]["frame"]) - 150) if mine else a
    cap = cv2.VideoCapture(video)
    n = -1
    while n < start - 1:
        cap.grab(); n += 1
    buf, pos = deque(maxlen=BUFFER), -1     # pos indexes buf; buf[-1] is the newest frame read
    paused, sp, warn = True, SPEEDS.index(1.0), ""
    pressed = []                            # frames marked this session, in key order, for undo
    win = "mark race phases"

    def read_next():
        nonlocal n
        if n >= b:
            return False
        ok, f = cap.read()
        if not ok:
            return False
        n += 1
        buf.append((n, cv2.resize(f, SHOW)))
        return True

    read_next(); pos = 0
    while True:
        vf, img = buf[pos]
        f = img.copy()
        clock = vf / FPS - offset
        lap = lap_of(clock)
        top = f"{pair}  {skater}  frame {vf}  clock {clock:7.1f} s  lap {lap if lap else '-'}  {'PAUSED' if paused else 'playing'} {SPEEDS[sp]}x"
        recent = "  ".join(f"{m['event']}{'' if m['seen'] else '?'}@{m['frame']}" for m in mine[-3:])
        bottom = f"next: {expected(mine)}   last: {recent}"
        keys = "[SPACE play  +/- speed  A/D frame  J/L 1 s  s start  e/x entry/exit  E/X not seen  f finish  U undo  Q quit]"
        for y, text in ((24, top), (SHOW[1] - 34, bottom), (SHOW[1] - 10, keys)):
            cv2.rectangle(f, (0, y - 20), (SHOW[0], y + 6), (0, 0, 0), -1)
            cv2.putText(f, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if any(m["frame"] == vf for m in mine):
            cv2.putText(f, "MARKED", (SHOW[0] - 170, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        if warn:
            cv2.putText(f, warn, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, f)
        k = cv2.waitKey(0 if paused else max(1, int(1000 / (FPS * SPEEDS[sp])))) & 0xFF
        if k == ord(" "):
            paused = not paused
        elif k in (ord("+"), ord("=")):
            sp = min(sp + 1, len(SPEEDS) - 1)
        elif k in (ord("-"), ord("_")):
            sp = max(sp - 1, 0)
        elif k in (ord("a"), ord("A"), ord("j"), ord("J")):
            paused = True
            pos = max(0, pos - (1 if k in (ord("a"), ord("A")) else int(FPS)))
            if pos == 0:
                warn = "start of the 10 s buffer"
        elif k in (ord("d"), ord("D"), ord("l"), ord("L")):
            paused, warn = True, ""
            for _ in range(1 if k in (ord("d"), ord("D")) else int(FPS)):
                if pos < len(buf) - 1:
                    pos += 1
                elif read_next():
                    pos = len(buf) - 1
        elif k in EVENTS:
            event, seen = EVENTS[k]
            want = expected(mine)
            warn = "" if event in want else f"marked {event}, expected {want}"
            mine = [m for m in mine if m["frame"] != vf]
            mine.append({"race": "beijing_2022_5000m", "pair": pair, "skater": skater, "frame": vf, "event": event,
                         "seen": seen, "clock_s": round(clock, 2), "lap": lap, "marked": datetime.now().strftime("%m/%d %H:%M")})
            mine.sort(key=lambda m: m["frame"])
            pressed = [p for p in pressed if p != vf] + [vf]
            save(others, mine)
        elif k in (ord("u"), ord("U")):
            if mine:
                gone = pressed.pop() if pressed else mine[-1]["frame"]
                mine = [m for m in mine if m["frame"] != gone]
                save(others, mine); warn = f"removed the mark at frame {gone}"
        elif k in (ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif not paused:
            if pos < len(buf) - 1:
                pos += 1
            elif read_next():
                pos = len(buf) - 1
            else:
                paused, warn = True, "end of the race window"
    cap.release()
    cv2.destroyAllWindows()
    save(others, mine)
    return mine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(WINDOWS))
    ap.add_argument("--skater", required=True)
    ap.add_argument("--from", dest="start", type=int)
    args = ap.parse_args()
    mine = run(args.pair, args.skater, args.start)
    print(len(mine), "marks for", args.skater)


if __name__ == "__main__":
    main()
