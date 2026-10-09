# Phase 6: Dual-Skater Tracking & Automatic Phase Detection

> Plan approved by the researcher 10/8 (tracker approach: `ultralytics`
> YOLO + ByteTrack; 6a target: identity correct on ≥95% of checked frames).

## Formal Research Question

> To what extent can a dedicated multi-object tracking approach reliably
> follow **both** skaters of a long-track pair through broadcast footage
> without identity swaps, and can corner and straightaway phases be detected
> **automatically** well enough that the Phase 5 form analysis runs on an
> unlabelled race video?

## Status (10/8): Started — 6a/6b first tracker run awaiting the researcher's identity check; 6d first improvements tested

## Why This Phase

Phase 5 showed the analysis methods work, but broadcast footage limited how
much of it could be used and required hand-labelling everything:

- **Roughly half of all candidate stretches were unusable** because both
  skaters were in frame (Inzell, Beijing pairs 3, 4, 7, 8; much of every
  race). The single-target tracker can't tell them apart reliably.
- **Every segment had to be hand-labelled** corner/straightaway (and camera
  shot type). The 5g classifier test reached 77% (target 85%), leaning on a
  camera-dependent feature.
- Dual tracking was attempted on 9/22 (three iterations) and failed on
  identity swaps. Diagnosed causes: ~21% duplicate MediaPipe detections, and
  similar-looking uniforms in different technique phases at the same moment.
  The diagnosis was that this needs a proper multi-object tracker, not more
  patches.

## What "Complete" Means Here

A defensible v1: both skaters tracked through a race with a **measured**
identity-swap rate against researcher-checked ground truth, and automatic
phase detection with a **measured** accuracy on skaters it never trained on.
Each result is reported whether it succeeds or not, with the threshold for
"good enough" fixed before testing.

---

## Tier 1 — Core

### 6a — Tracking Ground Truth & Benchmark
Before building anything, define how success is measured.
- Pick test clips with both skaters in frame (Beijing pairs, Inzell).
- Researcher marks which skater is which on sampled frames (frame sheets
  with numbered boxes) — the identity ground truth.
- Metrics: identity-swap count per minute, fraction of frames where each
  skater is correctly tracked, fraction of frames lost.
- Baseline: the current single-target tracker and the 9/22 dual attempts.
- **Target (to confirm with researcher):** identity correct on ≥95% of
  checked frames, with swaps rare enough to cut segments around them.

### 6b — Multi-Object Tracker
- A person detector that gives one clean box per skater (fixes the 9/22
  duplicate-detection problem), plus a tracker that uses motion (Kalman
  prediction) and appearance (suit colours) to keep identities.
- Options: an established detector + tracker (e.g. YOLO + ByteTrack via the
  `ultralytics` package — needs installing, small model download), or a
  custom Kalman + Hungarian-assignment tracker on top of MediaPipe (no new
  dependencies, but closer to what failed on 9/22).
- MediaPipe pose then runs separately on each tracked skater's box.

**First run (10/8):** installed `ultralytics` (8.4.174, YOLO11n). Built
`dual_track.py` (YOLO person detection + ByteTrack; boxes under 8% of frame
height dropped to skip distant crowd/officials) and ran it on 6 test clips
with both skaters in frame (`dual_tracking/test_clips.csv`): Beijing
Bergsma/Ichinohe, Bloemen/Zakharov (x2), Rijhnen/Lehman; Inzell
Eitrem/Jílek; Calgary Ghiotto/Eitrem start. Review sheets with each
track's ID drawn are in `dual_tracking/*_review.jpg` for the researcher's
identity check.
- **Preliminary look (Claude, not ground truth):** Rijhnen/Lehman tracked
  cleanly (IDs 1 and 2 throughout). Bergsma/Ichinohe close-up: when the two
  skaters overlap, the detector draws **one box around both** (e.g. 126312,
  126336, 126366), and after they separate the IDs can come back **swapped**
  (ID 93 on Bergsma at 126348, on Ichinohe by 126372). This pins the 9/22
  failure to a specific cause: overlapping skaters merge into one detection.
- Track counts per clip are high (8-40) because officials, coaches, crowd
  and camera cuts create extra and new IDs; what matters is whether the two
  skaters keep their own IDs, which the identity check measures.

### 6c — Two-Skater Extraction for Phase 5 Data
- Re-run the race scans with dual tracking to recover stretches previously
  skipped because both skaters were in frame.
- Researcher checks samples (as in Phase 5) before any are logged.
- Re-run 5f with the extra data — the most direct test of whether the
  trunk-asymmetry lead holds with more data.

### 6d — Automatic Corner/Straight Detection
- Start from the 5g classifier (77%) and the researcher's 163 labelled
  segments as training data.
- Improvements to test: longer time windows (a corner lasts ~10 s), the
  skater's movement across the frame and lean direction over time, and
  removing camera-dependent features that let it recognise shots instead of
  track sections.
- Evaluate leave-one-skater-out, same as 5g. **Target: ≥85% segment
  accuracy.**
- If reached: connect to `form_report.py` so a report can run on an
  unlabelled clip (completing what 5g couldn't).

**First improvements (10/8),** leave-one-skater-out, `phase_classifier.py`
with `PHASE_CLF_ANGLES_ONLY` / `PHASE_CLF_WIN`:

| Variant | Segment accuracy | Baseline |
|---|---|---|
| Original (5g) | 77.3% | 65.6% |
| Angles only, 1 s windows | 77.9% | 65.6% |
| Angles only, 2 s windows | 81.7% | 55.9%* |
| Whole segment, all signals | 80.2% | 64.7% |
| Whole segment, angles only | 79.0% | 64.7% |

\*2-second windows can't be cut from segments shorter than 2 s, so this is
tested on fewer segments — not directly comparable.
- Dropping the camera-dependent signals (projected hip width, pelvic tilt)
  did not hurt accuracy, and the most useful features became lean angle and
  its variation — physically what distinguishes a corner. A better basis,
  still short of the 85% target.
- Next to try: features describing a whole corner (~10 s) rather than short
  slices, e.g. sustained lean direction over time.

### 6e — Phase 6 Evaluation & Write-Up
Summary in the same format as Phase 5, including what didn't work.

---

## Tier 2 — If Time Allows

### 6f — Single-Camera 3D Pose Estimates
MediaPipe can output estimated 3D joint positions ("world landmarks") from
one camera; the pipeline doesn't capture them yet. Test whether 3D-estimated
angles are less affected by camera angle than 2D angles — e.g. whether the
5f trunk-asymmetry result holds up better within and across shot types.
Not true multi-camera 3D (see Future Work), but uses existing footage.

---

## Future Work (not this phase)
- **Controlled fixed-camera footage with a real fatigue measure** (lap times,
  heart rate) — the most direct fix for Phase 5's camera limitation.
- **True multi-camera 3D:** needs synchronized cameras; broadcast footage
  switches between cameras rather than showing one moment from several, and
  no public synchronized speed-skating dataset is known.
- **Inline skating** (Phase 5 Tier 3).

## Real Bugs & Limitations Found This Phase (Running Log)

| Date | Finding | Status |
|---|---|---|
| — | — | — |
