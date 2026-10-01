"""
Phase 5g: Unified Ice Form Report (first version, 9/29).

Ties 5a-5f into one report for one skater:
  5a  elite-anchor comparison per phase (prediction-adjusted t, p)
  5c  sit height (vertical, 2D, lateral hip-to-ankle)
  5d  stride rhythm per segment (only threshold-stable results trusted)
  5e  bilateral asymmetry (inside the 5a comparison, + corner vs straightaway)
  5f  early vs late race, if the skater has lap-labeled segments

Honesty rules built in:
  - The subject is ALWAYS compared against a reference built WITHOUT them
    (leave-one-out), so a reference member's report isn't flattered by their
    own data.
  - Every metric carries its known reliability caveat (see the running log in
    docs/CAPABILITIES_PHASE5.md). Arm swing (5b) is left out: its landmark
    misdetection problem is unresolved.
  - Sample sizes are printed next to every comparison.

Segments come from labeled_segments.csv. `external_test` rows need their real
phase given with --phase-override (e.g. Ragne Wiklund's is a straightaway).

Output: reports/<skater>_form_report.md

Usage:
    python -m form_report --skater "Patrick Meek"
    python -m form_report --skater "Sander Eitrem"
    python -m form_report --skater "Ragne Wiklund" --phase-override straightaway
"""

import os
import csv
import argparse
from datetime import date
import numpy as np
import pandas as pd
from scipy import stats

from compare_to_elite_reference import (
    LOG_PATH, VALID_PHASES, METRICS, extract_segment_means, get_segment_frames,
)
from stride_rhythm import analyze_segment
from validate_fatigue_form_degradation import LAP_RE, part_a_form, summarize_a

REPORT_DIR = "reports"
FLAG_P = 0.10
# With only 2 reference skaters, the reference SD comes from 2 values and can be
# tiny by chance (seen in the 9/28 leave-one-out validation), producing huge,
# meaningless scores -- so nothing is flagged below 3 reference skaters.
MIN_REF_FOR_FLAGS = 3

METRIC_LABELS = {
    "torso_lean_angle_deg": ("Torso lean", "deg"),
    "hip_velocity": ("Hip speed (on screen)", "torso/s"),
    "hip_acceleration_abs": ("Hip acceleration (on screen)", "torso/s²"),
    "right_knee_filtered": ("Sit height — right knee angle (main measure)", "deg"),
    "left_knee_filtered": ("Sit height — left knee angle (main measure)", "deg"),
    "hip_to_ankle_vertical_right": ("Sit height: vertical hip-ankle (secondary)", "torso"),
    "hip_to_ankle_2d_right": ("Sit height: 2D hip-ankle (secondary)", "torso"),
    "hip_to_ankle_lateral_right": ("Leg lateral extension", "torso"),
    "knee_angle_asymmetry": ("Knee asymmetry (L vs R)", "deg"),
    "hip_height_asymmetry": ("Pelvic tilt", "torso"),
    "torso_lean_lr_diff": ("Trunk lean L/R difference", "deg"),
    "hip_lateral_asymmetry": ("Projected hip width", "torso"),
}

# Known reliability problems, from the Phase 5 running log
CAVEATS = {
    "hip_velocity": "Measured on screen: not real speed when the camera tracks the skater (9/28).",
    "hip_acceleration_abs": "Measured on screen: not real acceleration when the camera tracks the skater (9/28).",
    "hip_lateral_asymmetry": "View-dependent: changes as the skater rotates relative to the camera (9/28).",
    "hip_to_ankle_vertical_right": ("Exaggerates changes: depends on a torso-based scale that shifts with camera "
                                    "viewpoint (9/30: Roest +38% while his knees straightened only 2-4%); wrong in "
                                    "start_rest with heavy torso foreshortening (9/28). Use the knee angles for sit height."),
    "hip_to_ankle_2d_right": ("Exaggerates changes, same cause as the vertical version (9/30). "
                              "Use the knee angles for sit height."),
}
# 9/30: metrics with a known reliability problem are still SHOWN, but never
# flagged as "unusual" -- a flag should only come from a metric we trust.
# Out-of-range values on these are listed separately, as things to check.


def load_rows():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reference_without(subject):
    """Per phase, per reference skater true means -- excluding the subject."""
    per_phase = {p: {} for p in VALID_PHASES}
    for row in load_rows():
        if row["phase"] not in VALID_PHASES or row["skater"] == subject:
            continue
        m = extract_segment_means(row)
        if m is not None:
            per_phase[row["phase"]].setdefault(row["skater"], []).append(m)
    ref = {}
    for phase, skaters in per_phase.items():
        means = {s: {k: np.mean([seg[k] for seg in segs if seg[k] is not None]) for k in METRICS}
                 for s, segs in skaters.items()}
        ref[phase] = pd.DataFrame(means).T  # rows = skaters
    return ref


def compare(subject_means, ref_df):
    n = len(ref_df)
    rows = []
    for m in METRICS:
        x = subject_means.get(m)
        if x is None or np.isnan(x) or n < 2:
            continue
        mu, sd = ref_df[m].mean(), ref_df[m].std(ddof=1)
        if not sd or np.isnan(sd):
            continue
        z = (x - mu) / sd
        t = z / np.sqrt(1 + 1 / n)
        p = 2 * stats.t.sf(abs(t), df=n - 1)
        rows.append({"metric": m, "you": x, "ref_mean": mu, "ref_sd": sd, "z": z, "t_pred": t, "p": p})
    return pd.DataFrame(rows), n


def fmt_metric(m):
    label, unit = METRIC_LABELS.get(m, (m, ""))
    return f"{label} ({unit})" if unit else label


def section_comparison(lines, phase, segs, ref):
    ref_df = ref.get(phase)
    means = [extract_segment_means(r) for r in segs]
    means = [m for m in means if m is not None]
    if not means:
        lines.append(f"_No usable {phase} data._\n")
        return None
    subject = {k: float(np.mean([m[k] for m in means if m[k] is not None])) for k in METRICS}
    if ref_df is None or len(ref_df) < 2:
        lines.append(f"_Reference has fewer than 2 other skaters for {phase} -- no comparison possible._\n")
        return subject
    cmp, n = compare(subject, ref_df)
    can_flag = n >= MIN_REF_FOR_FLAGS
    trusted = ~cmp["metric"].isin(list(CAVEATS))
    crit = stats.t.ppf(1 - FLAG_P / 2, df=n - 1)
    lines.append(f"Compared against **{n} other skaters** ({', '.join(ref_df.index)}), "
                 f"from {len(means)} of this skater's {phase} segment(s). "
                 + (f"A metric is flagged only if |t| > {crit:.2f} (p < {FLAG_P}), and only if it has no known "
                    f"reliability problem (⚠ metrics are shown but never flagged). With {int(trusted.sum())} "
                    f"flaggable metrics, about {FLAG_P * trusted.sum():.1f} flags are expected by chance alone.\n"
                    if can_flag else
                    f"**Too few reference skaters to flag anything** (need {MIN_REF_FOR_FLAGS}): with 2, the "
                    f"reference spread can be tiny by chance. Numbers shown for information only.\n"))
    lines.append("| Metric | This skater | Reference mean ± SD | t | p | |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in cmp.iterrows():
        caveat = r["metric"] in CAVEATS
        outside = can_flag and r["p"] < FLAG_P
        flag = ("outside range — ⚠ check" if caveat else "**unusual**") if outside else ""
        lines.append(f"| {fmt_metric(r['metric'])}{' ⚠' if caveat else ''} | {r['you']:.3f} | "
                     f"{r['ref_mean']:.3f} ± {r['ref_sd']:.3f} | {r['t_pred']:+.2f} | {r['p']:.2f} | {flag} |")
    outside = cmp[cmp["p"] < FLAG_P] if can_flag else cmp.iloc[0:0]
    flagged = outside[~outside["metric"].isin(list(CAVEATS))]
    unreliable = outside[outside["metric"].isin(list(CAVEATS))]
    lines.append("")
    if not can_flag:
        lines.append("")
        return subject
    if flagged.empty:
        lines.append(f"**Nothing stands out** on the trusted metrics for {phase}. With only {n} reference "
                     f"skaters, only very large differences can be detected.")
    else:
        lines.append(f"**Stands out:** {', '.join(fmt_metric(m) for m in flagged['metric'])}.")
    if not unreliable.empty:
        lines.append(f"Also outside the reference range, but on ⚠ metrics with a known reliability problem — "
                     f"check on video before reading anything into it: "
                     f"{', '.join(fmt_metric(m) for m in unreliable['metric'])}.")
    lines.append("")
    return subject


def section_rhythm(lines, segs):
    rows = []
    for r in segs:
        res, why = analyze_segment(r)
        seg = f"{r['start_frame']}-{r['end_frame']}"
        if res is None or why is not None:
            rows.append(f"| {r['phase']} | {seg} | — | — | — | not measurable: {why or 'no data'} |")
        else:
            trust = "stable" if res.get("stable") else "**threshold-sensitive, don't trust**"
            rows.append(f"| {r['phase']} | {seg} | {res['stride_period_s']:.2f} | {res['rhythm_cv']:.2f} | "
                        f"{res['lr_phase_offset']:.2f} | {trust} |")
    lines.append("| Phase | Frames | Stride period (s) | Rhythm CV | L/R timing | Reliability |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(rows)
    lines.append("\nRhythm CV: lower = more even strides (≈0.05 very even, >0.25 irregular). "
                 "L/R timing ≈0.5 = legs alternate evenly. Segments under ~2-3 s usually can't be measured.\n")


def section_fatigue(lines, segs):
    lap_segs = []
    for r in segs:
        m = LAP_RE.search(r.get("notes", ""))
        if m and r["phase"] in ("corner", "straightaway"):
            lap_segs.append({**r, "lap": int(m.group(1))})
    if not lap_segs:
        lines.append("_No lap-labeled full-race segments for this skater, so no early-vs-late comparison._\n")
        return
    df = pd.DataFrame(lap_segs)
    lo, hi = df["lap"].min(), df["lap"].max()
    third = (hi - lo) / 3
    df["stage"] = np.where(df["lap"] <= lo + third, "early", np.where(df["lap"] >= hi - third, "late", "middle"))
    df = df[df["stage"] != "middle"]
    a = summarize_a(part_a_form(df))
    if a.empty:
        lines.append("_Lap-labeled segments exist, but no phase has both early and late laps._\n")
        return
    lines.append(f"Early laps ≤ {lo + third:.0f}, late laps ≥ {hi - third:.0f}; same phase only. "
                 "Effect: +1 = every late frame higher, 0 = no difference.\n")
    lines.append("| Phase | Metric | Early | Late | Change | Effect |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in a.iterrows():
        if r["metric"] not in METRIC_LABELS and r["metric"] not in ("stride_period_s", "rhythm_cv"):
            continue
        name = fmt_metric(r["metric"]) if r["metric"] in METRIC_LABELS else r["metric"]
        eff = "" if np.isnan(r["late_vs_early_effect"]) else f"{r['late_vs_early_effect']:+.2f}"
        bold = "**" if abs(r["late_vs_early_effect"] or 0) >= 0.3 else ""
        lines.append(f"| {r['phase']} | {bold}{name}{bold} | {r['early']:.3f} | {r['late']:.3f} | "
                     f"{r['pct_change']:+.1f}% | {eff} |")
    lines.append("\nAssumes late race = more fatigued (a proxy, not measured fatigue). "
                 "The fatigue autoencoder (v2) is not reported here: on 9/28 a baseline control showed "
                 "no reliable early-vs-late signal from it yet.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skater", required=True)
    parser.add_argument("--phase-override", choices=VALID_PHASES,
                        help="phase to use for this skater's external_test rows")
    args = parser.parse_args()

    segs = []
    for r in load_rows():
        if r["skater"] != args.skater:
            continue
        if r["phase"] == "external_test" and args.phase_override:
            r = {**r, "phase": args.phase_override}
        if r["phase"] in VALID_PHASES:
            segs.append(r)
    if not segs:
        print(f"No usable segments for '{args.skater}'. (external_test rows need --phase-override.)")
        return

    print(f"Building reference without {args.skater} (leave-one-out)...")
    ref = reference_without(args.skater)

    lines = [f"# Form Report: {args.skater}", "",
             f"_Generated {date.today().isoformat()} by `form_report.py` (Phase 5g, first version). "
             f"{len(segs)} labeled segment(s)._", "",
             "> **Read with care.** Everything here comes from single-camera 2D broadcast video, and the "
             "elite reference has only a few skaters per phase. Treat findings as things to look at on "
             "video, not verdicts.", ""]

    lines += ["## 1. Compared with the elite reference (5a, 5c, 5e)", ""]
    for phase in VALID_PHASES:
        phase_segs = [r for r in segs if r["phase"] == phase]
        if phase_segs:
            lines += [f"### {phase.replace('_', ' ').title()}", ""]
            section_comparison(lines, phase, phase_segs, ref)

    lines += ["## 2. Stride rhythm (5d)", ""]
    section_rhythm(lines, [r for r in segs if r["phase"] in ("start_acceleration", "corner", "straightaway")])

    lines += ["## 3. Early vs late race (5f)", ""]
    section_fatigue(lines, segs)

    lines += ["## 4. Metric caveats", ""]
    for m, c in CAVEATS.items():
        lines.append(f"- ⚠ **{fmt_metric(m)}:** {c}")
    lines += ["- **Arm swing (5b)** is not reported: elbow landmarks can be confidently wrong (9/23).",
              "- **All metrics** are 2D projections from one camera; angles change with camera viewpoint.", ""]

    os.makedirs(REPORT_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in args.skater)
    path = os.path.join(REPORT_DIR, f"{safe}_form_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
