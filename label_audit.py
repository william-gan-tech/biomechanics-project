"""
Phase 9a (11/14): how close the older Phase 6 corner/straight labels are to
the frame-precise phase marks, to decide whether they can be used to train
the phase model.

Per frame, the marks give corner (entry to exit, including straight strokes
inside a corner) or straight (exit to entry, and start to the first entry).
Each Phase 6 piece labelled corner or straight for a marked skater is
compared frame by frame over its used frames (use_from-use_to, else the
whole piece).

Output: phase9/label_audit_<pair>.csv

Usage:
    python -m label_audit --pair pair10_vanderpoel
"""

import argparse
import pandas as pd

MARKS = "phase9/phase_marks.csv"
OLD = {"pair10_vanderpoel": "dual_tracking/beijing_p10_vanderpoel_swings_piece_labels.csv"}
NAMES = {"van der Poel": "Nils van der Poel", "Swings": "Bart Swings"}


def frame_labels(marks):
    """Frame -> 'corner' / 'straight' between the skater's start and finish (or last usable mark)."""
    lab, state, prev = {}, None, None
    for _, m in marks.sort_values("frame").iterrows():
        if prev is not None and state is not None:
            for f in range(prev, int(m.frame)):
                lab[f] = state
        state = {"start": "straight", "entry": "corner", "exit": "straight", "in_straight": "corner",
                 "in_cross": "corner", "finish": None}[m.event]
        prev = int(m.frame)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(OLD))
    args = ap.parse_args()
    marks = pd.read_csv(MARKS)
    marks = marks[(marks.pair == args.pair) & (marks.use == "yes")]
    old = pd.read_csv(OLD[args.pair])
    old = old[old.phase.isin(["corner", "straight"]) & old.skater.isin(NAMES)]
    rows = []
    for _, p in old.iterrows():
        sk = NAMES[p.skater]
        lab = frame_labels(marks[marks.skater == sk])
        a = int(p.use_from) if pd.notna(p.use_from) else int(p.first_frame)
        b = int(p.use_to) if pd.notna(p.use_to) else int(p.last_frame)
        frames = [f for f in range(a, b + 1) if f in lab]
        if not frames:
            continue
        agree = sum(lab[f] == p.phase for f in frames)
        rows.append({"piece_no": p.piece_no, "skater": sk, "from": a, "to": b, "old_label": p.phase,
                     "frames": len(frames), "agree": agree, "agree_share": round(agree / len(frames), 3),
                     "covered_by_marks": round(len(frames) / (b - a + 1), 3)})
    r = pd.DataFrame(rows)
    r.to_csv(f"phase9/label_audit_{args.pair}.csv", index=False)
    print(r.to_string(index=False))
    print("pieces", len(r), "fully agreeing", (r.agree_share == 1).sum(),
          "frames agreeing", f"{r.agree.sum()}/{r.frames.sum()} = {r.agree.sum() / r.frames.sum():.1%}")


if __name__ == "__main__":
    main()
