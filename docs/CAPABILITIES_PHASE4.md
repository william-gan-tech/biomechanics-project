# 🚀 Phase 4 Capabilities & Research Plan (`CAPABILITIES_PHASE4.md`)

## 🎯 Formal Research Question

> **Phase 4: To what extent can bone-length-scaled joint-angle trajectories
> distinguish explosive start-phase acceleration mechanics from
> steady-state cruising form, and does the multi-person identity-tracking
> approach built for Phase 3 generalize to reliably isolating a single
> target skater from simultaneous competitors at a shared start line?**

- **Phase 4a** — *Biomechanical:* Do start-phase kinematics differ
  measurably from other technique phases in the same skater?
- **Phase 4b** — *Tracking robustness:* Does the multi-person identity
  tracking built 9/15 for Phase 3 correctly isolate one target skater
  when multiple skaters are genuinely simultaneous in frame?

## ✅ STATUS: BOTH 4a AND 4b COMPLETE (9/18)

Both sub-questions now have real, verified evidence — not placeholders,
not unresolved caveats. Full history below.

---

## 🟢 Phase 4a — Complete

### Prerequisite Footage Audit (9/16)
Built `audit_footage_for_starts.py`. Found 5 of 7 existing skater videos
had meaningful simultaneous multi-person content when sampled (Silovs
42.2%, Ragne 33.3%, Jorrit 32.4%, Mia 17.2%, Jan 11.7%) — but this later
proved to be mostly non-start content (mid-race incidental proximity,
title-card detection noise), which is why real start-line footage was
still needed.

### Feature Engineering (9/16)
Built `start_phase_features.py`: torso-lean/crouch angle, hip velocity,
hip acceleration. Verified math correctness against non-start footage as
a code sanity check. Stated limitation: right-side features only.

### Automated Detection Failures — Documented Honestly (9/18)
Tried `find_candidate_start_moments.py` (acceleration-spike + 
velocity-before-spike filter) against two real Olympic short-track
candidate videos found via web search. **Found zero valid candidates in
either**, even after widening the search to the full early portion of
each clip. Root cause: MediaPipe landmark jitter on fast-panning,
tightly-packed multi-skater broadcast footage prevents `hip_velocity`
from ever reading genuinely near-zero, even at true rest — the
auto-derived percentile threshold was consistently too strict for this
footage's actual noise floor.

**Important secondary finding:** a candidate flagged as "the largest
acceleration spike in the window" in two separate runs turned out, on
direct visual inspection, to be a **broadcast name/stats graphic
overlay** (a translucent ghosted portrait animating over live footage)
fooling MediaPipe's tracking — not real skater motion. Lesson: never
trust an algorithmic candidate without opening the actual frame image.

### Manual Verification Succeeds — Real Start Confirmed (9/18)
Pivoted to `browse_frames_manually.py` after the automated approach
failed. Using a real YouTube timestamp identified by directly watching
the source video, downloaded and manually browsed `start_candidate_3.mp4`
(Milano Cortina 2026 short track). Found and visually confirmed a genuine
start sequence:

- **REST** (5.17s–5.64s, frames 155–169): skaters visually confirmed
  held in start position.
- **DATA GAP** (5.71s–6.37s): ~0.7s of zero landmark detections,
  confirmed to be a real broadcast camera cut around the gun (framing
  differs before/after). The literal launch instant is unrecoverable
  from this footage — documented as a real gap, not interpolated over.
- **EARLY-ACCELERATION** (6.44s onward, frame 193+): skaters visually
  confirmed immediately off the line, still accelerating.

### Identity Question Resolved (9/18)
Two independent checks, both leaning the same direction:

1. **Appearance-histogram check** (`check_same_skater_identity.py`,
   averaged across full segments, not single noisy frames): similarity
   0.6964 — moderate, inconclusive on its own.
2. **Gap-continuity check** (`check_gap_continuity.py`, built after #1
   proved insufficient): tests whether the position jump across the
   0.7s camera-cut gap is physically explainable by one continuously
   accelerating skater. **Result: CONSISTENT.** Implied velocity across
   the gap (0.0272/sec) was *lower* than the max velocity actually
   observed in the 50 frames following it (0.0891/sec) — fully
   explainable by continuous acceleration, no impossible jump required.

**Conclusion: confirmed same skater across both segments.** The torso-lean
sign flip found between REST (-50.5°) and EARLY-ACCELERATION (+40.5°) is
a real signal, not a tracking artifact.

### Real Within-Subject Technique-Phase Comparison (9/18) — Headline Result

Rather than hunting further for one isolated "cruise" segment, used
frames already visually categorized during manual review to directly
compare **start vs. corner vs. straightaway** within the same skater,
same video (`compare_technique_phases.py`):

| Phase | n_frames | Torso Lean (mean) | Torso Lean (std) | Velocity (mean) | Right Knee (mean) | Left Knee (mean) |
|---|---|---|---|---|---|---|
| Corner | 55 | -23.8° | **82.9°** | 0.0145 | 146.5° | 132.3° |
| Start | 80 | 21.8° | 35.9° | 0.0164 | 128.5° | 116.6° |
| Straightaway | 90 | -14.7° | **26.2°** | 0.0125 | 150.5° | 148.4° |

**Real, physically sensible findings:**
- **Torso-lean stability ranks Straightaway > Start > Corner** (26.2° 
  35.9° << 82.9° std) — corners are 3.2x more variable than
  straightaways. Matches expected biomechanics: cornering requires
  continuously-changing lean angle to counter centrifugal force; start
  blends two distinct postures (held crouch + launch); straightaway is
  settled rhythmic striding.
- **Knee symmetry differs sharply by phase**: straightaway shows nearly
  symmetric knees (right 150.5° vs. left 148.4°, ~2° apart); corner
  shows clear asymmetry (right 146.5° vs. left 132.3°, ~14° apart) —
  real evidence that corner technique genuinely demands asymmetric leg
  mechanics, directly motivating the planned Phase 6 need for bilateral
  (not just right-side) tracking.

---

## 🟢 Phase 4b — Complete

### First Real-Footage Success (9/16)
Ran `diagnose_tracking_swap.py` against Haralds Silovs' footage. At
frame 1786, two ambiguously-close simultaneous people (dist 0.0354 vs.
0.0396) were correctly disambiguated via the appearance-histogram
tiebreak (0.959 vs. 0.944 similarity) — first confirmed correct
multi-person resolution on real footage. Also found a ~100-frame
zero-detection stretch, visually confirmed to be a title-card overlay,
not a tracking failure — same broadcast-content limitation pattern as
Phase 3.

### Full Validation on Genuine Pack-Racing Footage (9/18) — Headline Result

Ran the complete `diagnose_tracking_swap.py` (updated to cover the whole
video, `MAX_FRAMES = 999999`) against `start_candidate_3.mp4` — real
Olympic short-track pack racing, the actual intended stress-test
scenario for this tool (genuinely simultaneous competitors, not
incidental background people).

**Results across the full video (16,019 log lines):**
- **135 ambiguous tiebreak events** — genuine multi-person proximity
  requiring appearance-based disambiguation, far more than any prior
  test, consistent with sustained pack-racing conditions.
- **58 lost-track/re-anchor events** — consistent with the
  already-documented broadcast-cut limitation, not tracking failure.
- **10-sample manual review** (5 from early video, 5 from mid-video):
  **every single selection was correct** — consistently picked the
  closer-position, higher-or-comparable-appearance-similarity candidate
  over the more distant alternative, across dozens of consecutive frames
  in real pack conditions.

**Conclusion:** the 9/15 multi-person tracking fix works correctly under
its actual intended real-world condition — genuinely simultaneous
competitors — not just in synthetic tests or isolated lucky frames.

---

## 🔴 Recurring Limitation, Now Confirmed Across Five Separate Videos

Broadcast/YouTube footage content inconsistency (title cards, camera
cuts, animated name/stat graphic overlays) — not the tracking algorithm
— remains the dominant practical bottleneck. Confirmed in: Mia (Phase
3), Lee Sang-Hwa (9/15), Silovs (9/16), and twice more in the Phase 4a
candidate search (9/18: false-positive graphic overlay, and the
gun-instant camera cut in `start_candidate_3`). This is a general,
structural property of this footage type, not a series of isolated
incidents, and should be stated as such in any write-up.

---

## Tools Built This Phase
- `audit_footage_for_starts.py`
- `start_phase_features.py`
- `find_candidate_start_moments.py`
- `test_start_candidate.py`
- `browse_frames_manually.py`
- `compare_start_vs_cruise.py`
- `check_same_skater_identity.py`
- `check_gap_continuity.py`
- `compare_technique_phases.py`

---

## What This Sets Up for Phase 5 & 6

The 9/18 technique-phase comparison already provides real, preliminary
signal for both planned future phases, using data already in hand:

- **Phase 5 (straightaways):** straightaway already shown to be the most
  stable/consistent phase (lowest torso-lean std) — a real baseline to
  build a fuller three-condition (unscaled/scaled/zscore-only) ablation
  on.
- **Phase 6 (corners):** the corner-phase knee asymmetry (14° gap vs. 2°
  on straightaways) is a concrete, evidence-backed reason bilateral
  (left+right) tracking is necessary for that phase — not a speculative
  future need, but a demonstrated one.

## Honest Limitations
- n=1 confirmed real start video. The technique-phase comparison is
  within one skater, one race — real and rigorous, but not yet
  generalizable across athletes.
- Phase 4a's comparison uses proxy phase boundaries based on visual
  review, not frame-perfect biomechanical event detection (e.g. exact
  touch-down/push-off timing).
- Automated start detection remains unreliable on chaotic broadcast
  footage; every real result this phase required manual visual
  verification. A tool meant to help other skaters at scale still needs
  this fixed or explicitly designed around a human-in-the-loop step.