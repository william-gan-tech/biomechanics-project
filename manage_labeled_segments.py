"""
Sets up and manages a structured labeling log for confirmed technique-phase
segments across skaters -- the actual infrastructure needed to build a
reference dataset, instead of hardcoding frame ranges into a new script
every time.

Usage:
    python -m manage_labeled_segments --init
    python -m manage_labeled_segments --add --skater "Sven Kramer" --video "data/sven_kramer_ref.mp4" --phase straightaway --start 400 --end 460 --notes "visually confirmed steady stride"
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
    elif args.list:
        list_segments()
    elif args.summary:
        summary()
    else:
        print("Specify one of: --init, --add, --list, --summary. See docstring for usage examples.")


if __name__ == "__main__":
    main()
