"""
Sets up and manages a structured labeling log for confirmed technique-phase
segments across skaters -- the actual infrastructure needed to build a
reference dataset, instead of hardcoding frame ranges into a new script
every time.

Usage:
    python -m manage_labeled_segments --init
    python -m manage_labeled_segments --add --skater "Sven Kramer" --video "data/sven_kramer_ref.mp4" --phase straightaway --start 400 --end 460 --notes "visually confirmed steady stride"
    python -m manage_labeled_segments --remove --skater "Sven Kramer" --start 400 --end 460
    python -m manage_labeled_segments --list
    python -m manage_labeled_segments --summary
"""

import os
import csv
import argparse

LOG_PATH = "labeled_segments.csv"
FIELDNAMES = ["skater", "video_path", "phase", "start_frame", "end_frame", "date_confirmed", "notes"]
VALID_PHASES = ["start_rest", "start_acceleration", "corner", "straightaway"]


def init_log():
    if os.path.exists(LOG_PATH):
        print(f"{LOG_PATH} already exists -- not overwriting. Delete it manually first if you want a fresh start.")
        return
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
    print(f"Initialized {LOG_PATH}")


def add_segment(skater, video_path, phase, start_frame, end_frame, notes, date_confirmed):
    if phase not in VALID_PHASES:
        print(f"WARNING: '{phase}' is not in the standard phase list {VALID_PHASES}. "
              f"Adding anyway, but consider using a standard name for consistency.")

    if not os.path.exists(LOG_PATH):
        init_log()

    # FIXED 9/28: if the file doesn't end with a newline (e.g. after a hand
    # edit), appending glues the new row onto the last one -- this silently
    # swallowed rows twice (Patrick Meek corner 9/21, Eitrem lap 2 9/28).
    with open(LOG_PATH, "rb") as f:
        data = f.read()
    if data and not data.endswith((b"\n", b"\r")):
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            f.write("\r\n")

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({
            "skater": skater,
            "video_path": video_path,
            "phase": phase,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "date_confirmed": date_confirmed,
            "notes": notes,
        })
    print(f"Added: {skater} | {phase} | frames {start_frame}-{end_frame} | {video_path}")


def remove_segment(skater, start_frame, end_frame, video_path=None):
    """Removes rows matching skater + start + end (+ video if given). Refuses
    to remove more than one row unless they are exact duplicates, so a typo
    can't wipe out several segments at once."""
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def matches(r):
        return (r["skater"] == skater and r["start_frame"] == str(start_frame)
                and r["end_frame"] == str(end_frame)
                and (video_path is None or r["video_path"] == video_path))

    hits = [r for r in rows if matches(r)]
    if not hits:
        print(f"No row found for {skater} frames {start_frame}-{end_frame}. Nothing removed.")
        print("Use --list to see exact skater names and frame ranges.")
        return
    if len(hits) > 1 and len({tuple(r.values()) for r in hits}) > 1:
        print(f"{len(hits)} different rows match {skater} {start_frame}-{end_frame} -- add --video to pick one. Nothing removed.")
        return

    kept = [r for r in rows if not matches(r)]
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)
    for r in hits:
        print(f"Removed: {r['skater']} | {r['phase']} | frames {r['start_frame']}-{r['end_frame']} | {r['video_path']}")


def list_segments():
    if not os.path.exists(LOG_PATH):
        print(f"{LOG_PATH} doesn't exist yet -- run with --init first.")
        return
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("No segments logged yet.")
        return
    for row in rows:
        print(f"  {row['skater']:25s} | {row['phase']:20s} | frames {row['start_frame']}-{row['end_frame']:5s} | {row['video_path']}")
    print(f"\nTotal: {len(rows)} segments")


def summary():
    if not os.path.exists(LOG_PATH):
        print(f"{LOG_PATH} doesn't exist yet -- run with --init first.")
        return
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No segments logged yet.")
        return

    by_phase = {}
    for row in rows:
        by_phase.setdefault(row["phase"], set()).add(row["skater"])

    print("=" * 60)
    print("REFERENCE DATASET COVERAGE SUMMARY")
    print("=" * 60)
    for phase in VALID_PHASES:
        skaters = by_phase.get(phase, set())
        status = "GOOD sample size" if len(skaters) >= 5 else ("growing" if len(skaters) >= 2 else "needs more data")
        print(f"{phase:20s}: {len(skaters)} skater(s) -- {status}")
        if skaters:
            print(f"    {', '.join(sorted(skaters))}")

    other_phases = set(by_phase.keys()) - set(VALID_PHASES)
    if other_phases:
        print(f"\nOther/custom phases logged: {sorted(other_phases)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Initialize a new empty log")
    parser.add_argument("--add", action="store_true", help="Add a new segment")
    parser.add_argument("--remove", action="store_true", help="Remove a segment by --skater --start --end")
    parser.add_argument("--list", action="store_true", help="List all logged segments")
    parser.add_argument("--summary", action="store_true", help="Show coverage summary by phase")
    parser.add_argument("--skater", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--phase", type=str, choices=VALID_PHASES + ["custom"])
    parser.add_argument("--custom-phase", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    args = parser.parse_args()

    if args.init:
        init_log()
    elif args.add:
        phase = args.custom_phase if args.phase == "custom" else args.phase
        if not all([args.skater, args.video, phase, args.start is not None, args.end is not None]):
            print("--add requires --skater, --video, --phase, --start, --end")
            return
        add_segment(args.skater, args.video, phase, args.start, args.end, args.notes, args.date or "unspecified")
    elif args.remove:
        if not all([args.skater, args.start is not None, args.end is not None]):
            print("--remove requires --skater, --start, --end (and optionally --video)")
            return
        remove_segment(args.skater, args.start, args.end, args.video)
    elif args.list:
        list_segments()
    elif args.summary:
        summary()
    else:
        print("Specify one of: --init, --add, --list, --summary. See docstring for usage examples.")


if __name__ == "__main__":
    main()
