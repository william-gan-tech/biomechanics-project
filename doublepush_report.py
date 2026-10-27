"""
Phase 7d (10/26): double-push technique feedback report.

A clip known to show double push goes in; each technique measure is compared
with the elite reference (Joey Mantia and Viktor Hald Thorup's good double
push), and anything outside the elite range is explained with the
researcher's coaching text (doublepush/feedback_text.json).

Measures per 4-second window (front or back view, leg >= 150 px):
  no_underpush    cross_frac: share of frames with a foot past the hip midline
                  (flag if below the elite 5th percentile)
  wide_setdown    setdown: sideways landing position of the recovering foot,
                  leg lengths, + = out to its own side (flag above 95th)
  jerky_path      foot_jerk: roughness of the feet's sideways paths (above 95th)
  hip_bounce      hip_bob: up-down movement of the hips (above 95th)
  upper_body      upper_sway: shoulders moving relative to the hips; only when
                  the shoulders are visible (above 95th)
  no_fall_catch   fall_catch: how far the hips shift toward the landing foot in
                  the 0.3 s before it touches down, leg lengths (below 5th)

Rules fixed 10/26 before testing:
- Flag thresholds: elite 5th/95th percentiles of the reference windows.
- A clip's report lists an issue when it is flagged in at least half of the
  clip's judgeable windows.
- Slow-motion clips: time-based measures (jerky_path, hip_bounce,
  no_fall_catch) are not judged, since playback speed changes them; spatial
  measures are.
Test:
- Mantia 3:03-3:09 (pressure spike, wild movement) should get jerky_path or
  hip_bounce.
- Mantia 3:15-3:23 (too much upper body) should get upper_body, or say upper
  body can't be judged.
- New elite skaters (Lars Scheenstra, Mark Horsten; never in the reference):
  each measure flagged in at most 10% of windows.

Usage:
    python -m doublepush_report             # build reference, run the test, write reports
"""

import os
import json
import numpy as np
import pandas as pd
import cv2

from doublepush_measures import track, measures, MIN_LEG_PX, WINDOW_S, MIN_TRACKED
from recovery_measure import touchdowns

FEEDBACK = "doublepush/feedback_text.json"
OUT_DIR = "reports/doublepush"
LOW = 0.05
HIGH = 0.95
# measure -> (column, direction of a problem)
ISSUES = {"no_underpush": ("cross_frac", "low"), "wide_setdown": ("setdown", "high"),
          "jerky_path": ("foot_jerk", "high"), "hip_bounce": ("hip_bob", "high"),
          "upper_body": ("upper_sway", "high"), "no_fall_catch": ("fall_catch", "low")}
TIME_BASED = {"jerky_path", "hip_bounce", "no_fall_catch"}

REFERENCE = [  # (labels file, frames prefix, which rows)
    ("doublepush/style_labels.csv", "doublepush/frames_", lambda r: r.label == "double_push" and r.skater_note.startswith("Mantia tutorial")),
    ("doublepush/fresh_test_labels.csv", "doublepush/fresh_frames_", lambda r: r.label == "double_push"),
]
TESTS = [
    ("Mantia flawed 3:03-3:09", "doublepush/style_labels.csv", "doublepush/frames_", lambda r: r.skater_note == "Mantia 3:03-3:09", False),
    ("Mantia flawed 3:15-3:23", "doublepush/style_labels.csv", "doublepush/frames_", lambda r: r.skater_note == "Mantia 3:15-3:23", False),
    ("Lars Scheenstra (slow motion)", "doublepush/new_test_labels.csv", "doublepush/new_frames_", lambda r: r.label == "double_push", True),
    ("Mark Horsten (slow motion)", "doublepush/report_test_labels.csv", "doublepush/report_frames_", lambda r: True, True),
]


def fall_catch(d, t, fps):
    s = d.leg_px.median()
    side = np.sign((d.lf / s).median() - (d.rf / s).median()) or 1.0
    lag = max(1, int(0.3 * fps))
    idx = d.set_index("frame")
    vals = []
    for r in t.itertuples():
        a = r.frame - lag
        if a in idx.index and r.frame in idx.index:
            toward = side if r.foot == "l" else -side
            vals.append(toward * (idx.hip_x[r.frame] - idx.hip_x[a]) / s)
    return float(np.median(vals)) if vals else np.nan


def window_measures(path, prefix, pick, track_missing=False):
    lab = pd.read_csv(path)
    rows = []
    for n, r in enumerate(lab.itertuples()):
        if not pick(r):
            continue
        cache = f"{prefix}{n:02d}.csv"
        fps = cv2.VideoCapture(r.video).get(cv2.CAP_PROP_FPS) or 30.0
        if os.path.exists(cache) and "lf_y" in pd.read_csv(cache, nrows=1).columns:
            d = pd.read_csv(cache)
        elif track_missing:
            d, fps = track(r.video, int(r.start_frame), int(r.end_frame)); d.to_csv(cache, index=False)
        else:
            continue
        a0, a1 = int(r.start_frame), int(r.end_frame)
        w = int(WINDOW_S * fps)
        starts = [a0] if a1 - a0 + 1 <= 1.5 * w else list(range(a0, a1 - w + 2, w))
        for k, a in enumerate(starts):
            b = a1 if len(starts) == 1 else a + w - 1
            g = d[(d.frame >= a) & (d.frame <= b)]
            if len(g) < 15 or len(g) < MIN_TRACKED * (b - a + 1) or g.leg_px.median() < MIN_LEG_PX:
                continue
            m = measures(g, fps)
            t = touchdowns(g, fps)
            rows.append({"stretch": r.skater_note, "window": k, "start_s": a / fps,
                         "cross_frac": m["cross_frac"], "foot_jerk": m["foot_jerk"], "hip_bob": m["hip_bob"],
                         "upper_sway": m["upper_sway"],
                         "setdown": t.setdown.median() if len(t) >= 2 else np.nan,
                         "fall_catch": fall_catch(g, t, fps) if len(t) >= 2 else np.nan})
    return pd.DataFrame(rows)


def flags(w, ref_q, slow):
    out = {}
    for issue, (col, direction) in ISSUES.items():
        v = w[col]
        if (slow and issue in TIME_BASED) or pd.isna(v):
            out[issue] = None                      # can't judge
        elif direction == "high":
            out[issue] = bool(v > ref_q[col][HIGH])
        else:
            out[issue] = bool(v < ref_q[col][LOW])
    return out


def write_report(name, win, ref_q, slow, text):
    lines = [f"# Double-push feedback: {name}", "",
             f"{len(win)} four-second windows judged. Compared with the elite reference "
             "(Joey Mantia and Viktor Hald Thorup). An issue is listed when it shows up in at least half of the windows.", ""]
    listed, unjudged = [], []
    for issue in ISSUES:
        f = [x for x in win[issue] if x is not None]
        if not f:
            unjudged.append(issue)
        elif np.mean(f) >= 0.5:
            listed.append((issue, np.mean(f)))
    if listed:
        lines.append("## What to work on")
        for issue, share in listed:
            t = text[issue]
            lines += ["", f"### {t['title']}", f"*Seen in {share:.0%} of windows.*", "", t["explain"], "",
                      f"- **Cue:** {t['cue']}", f"- **Drill:** {t['drill']}"]
    else:
        lines.append("## No issues outside the elite range")
    if unjudged:
        lines += ["", "## Not judged on this clip", ""]
        for issue in unjudged:
            why = "slow motion changes timing-based measures" if (slow and issue in TIME_BASED) else "not visible in this footage"
            lines.append(f"- {text[issue]['title']} ({why})")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") + ".md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return listed, unjudged, path


def main():
    text = json.load(open(FEEDBACK, encoding="utf-8"))
    ref = pd.concat([window_measures(p, f, pick) for p, f, pick in REFERENCE], ignore_index=True)
    ref_q = {col: ref[col].quantile([LOW, HIGH]) for col, _ in ISSUES.values()}
    print(f"Elite reference: {len(ref)} windows ({ref.stretch.str.startswith('Mantia').sum()} Mantia, "
          f"{ref.stretch.str.startswith('Thorup').sum()} Thorup)")
    for col, _ in ISSUES.values():
        print(f"  {col:11s} 5th {ref_q[col][LOW]:8.3f}   95th {ref_q[col][HIGH]:8.3f}")
    print()
    for name, path, prefix, pick, slow in TESTS:
        if not os.path.exists(path):
            print(f"{name}: labels missing, skipped"); continue
        w = window_measures(path, prefix, pick, track_missing=True)
        if w.empty:
            print(f"{name}: no judgeable windows"); continue
        fl = pd.DataFrame([flags(r, ref_q, slow) for _, r in w.iterrows()])
        listed, unjudged, rp = write_report(name, fl, ref_q, slow, text)
        rates = {k: (np.mean([x for x in fl[k] if x is not None]) if any(x is not None for x in fl[k]) else None) for k in ISSUES}
        print(f"{name}: {len(w)} windows -> {rp}")
        print("  flag rate per measure: " + ", ".join(f"{k} {'n/a' if v is None else f'{v:.0%}'}" for k, v in rates.items()))
        print(f"  report lists: {[i for i, _ in listed] or 'nothing'}; not judged: {unjudged}")


if __name__ == "__main__":
    main()
