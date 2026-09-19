# 🚀 Phase 4 Capabilities & Research Plan (`CAPABILITIES_PHASE4.md`)

## 🎯 Formal Research Question

> **Phase 4: To what extent can bone-length-scaled joint-angle trajectories
> distinguish explosive start-phase acceleration mechanics from
> steady-state cruising form, and does the multi-person identity-tracking
> approach built for Phase 3 generalize to reliably isolating a single
> target skater from simultaneous competitors at a shared start line?**

This is two connected sub-questions, not one — split for the same reason
Phase 3 was split into 3a/3b: precision about what each experiment
actually tests.

- **Phase 4a** — *Biomechanical:* Do start-phase kinematics (crouch angle,
  initial push-off explosiveness, acceleration curve) differ measurably
  from steady-state cruising form in the same skater, using the existing
  bone-scaled feature pipeline?
- **Phase 4b** — *Tracking robustness:* Does the multi-person identity
  tracking built 9/15 for Phase 3 (position continuity + appearance
  histogram) correctly isolate one target skater when multiple skaters are
  **genuinely simultaneous** in frame (a start line), as opposed to the
  **sequential compilation cuts** it was tested against and found to fail
  on (Lee Sang-Hwa's video)?

---

## 🟢 Status: Active — Real Data on Both Sub-Questions (9/16–9/18)

### Prerequisite Footage Audit — Complete (9/16)
Built `audit_footage_for_starts.py` to check whether existing footage
already contains genuine simultaneous multi-person content, before
assuming new video needed to be sourced.

**Result:** 5 of 7 existing skater videos show meaningful simultaneous
multi-person content when sampled: Haralds Silovs (42.2%), Ragne Wiklund
(33.3%), Jorrit Bergsma (32.4%), Mia Manganello Kilburg (17.2%), Jan
Blokhuijsen (11.7%).

### 4a — Feature Engineering: Built, Verified, Now With Real Start Data

Built `start_phase_features.py` computing three new features not present
in the Phase 3 feature set: torso-lean/crouch angle, hip velocity, hip
acceleration (right-side only — a stated, unresolved limitation given
skating's push-off asymmetry).

**9/16 — Math verified** against non-start footage (Sven Kramer's
reference video) as a code sanity check only; correctly produced a
pattern that did NOT resemble a real start, confirming the math isn't
spuriously detecting patterns that aren't there.

**9/18 — First real, visually-confirmed start-phase data point.**
Automated acceleration-spike detection (`find_candidate_start_moments.py`)
was tried against two real Olympic short-track candidate videos and
**found zero valid candidates in either**, even after widening the search
to the entire early portion of each clip (see "What Didn't Work" below).
Pivoted to manual visual frame browsing (`browse_frames_manually.py`) on
a third candidate — `start_candidate_3.mp4` (Milano Cortina 2026, short
track), downloaded from a YouTube timestamp the user identified by
directly watching the source video — and found a genuine start sequence,
confirmed by eye, not by algorithm.

**Confirmed segment structure:**
- **REST** (5.17s–5.64s, frames 155–169): skaters visually confirmed held
  in start position.
- **DATA GAP** (5.71s–6.37s, ~0.7s): zero landmark detections. Visually
  confirmed to be a real broadcast camera cut around the gun (framing
  differs before/after) — the literal launch instant is unrecoverable
  from this footage. Documented honestly as a gap, not interpolated over.
- **EARLY-ACCELERATION** (6.44s onward, frame 193+): skaters visually
  confirmed immediately off the line, still accelerating.

**Result** (`compare_start_vs_cruise.py`):

| Segment | Torso Lean (mean) | Velocity (mean) | Acceleration (mean) |
|---|---|---|---|
| REST | -50.5° (±23.1) | 0.0081 | 0.0053 |
| EARLY-ACCELERATION | +40.5° (±21.1) | 0.0196 | 0.0117 |

Velocity is **2.4x higher** during early-acceleration than rest — the
first real, directionally-correct quantitative signal for the Phase 4a
hypothesis.

**Two open issues, not yet resolved:**
1. Torso lean **flips sign entirely** between segments (-50° → +40°),
   possibly indicating the multi-person tracker locked onto a *different*
   skater between the two segments rather than reflecting a real
   body-mechanics change in one athlete. Needs a same-identity check
   before this number is trusted.
2. The default cruise-comparison window (16–20s) was visually checked and
   found to still show highly volatile torso-lean/velocity values —
   evidently still transitional pack content, not settled steady-state
   cruising. A later window needs to be found and confirmed, or a
   long-track clip may be needed instead, since short-track races may
   simply be too short to contain a genuine cruise phase at all.

### 4b — Tracking Test on Real Footage: Confirmed Success + Recurring Limitation

Ran `diagnose_tracking_swap.py` against Haralds Silovs' footage (9/16).

**Real success:** At frame 1786, two genuinely simultaneous people were
detected with ambiguous position distances (0.0354 vs. 0.0396). The
appearance-histogram tiebreak correctly activated and selected the
higher-similarity candidate (0.959 vs. 0.944 correlation) — the tracking
system working exactly as designed, on real footage.

**Recurring limitation, now confirmed across four separate videos** (Mia
in Phase 3, Lee Sang-Hwa 9/15, Silovs 9/16, and the broadcast-graphic
false positive found 9/18 on `start_candidate_1`/`start_candidate_3`):
broadcast content (title cards, cuts, animated name/stats overlays) is
the dominant practical bottleneck, not the tracking algorithm itself.
The 9/18 finding specifically showed that even a *plausible-looking
acceleration spike* can be a broadcast graphic overlay fooling the
tracker, not real motion — meaning algorithmic detection results on this
class of footage cannot be trusted without visual confirmation of the
underlying frames, a new and important addition to this limitation
pattern.

---

## What Didn't Work (9/18) — Documented Honestly, Not Discarded

- Automated velocity-before-spike filtering found **zero valid start
  candidates** across two real Olympic candidate videos, despite finding
  many acceleration spikes. Root cause: MediaPipe landmark jitter on
  fast-panning, tightly-packed multi-skater broadcast footage prevents
  `hip_velocity` from ever reading genuinely near-zero, even at true
  rest — the auto-derived percentile threshold was consistently too
  strict for this footage's noise floor.
- A candidate flagged as "the largest acceleration spike in the window"
  in two separate runs turned out, on direct visual inspection, to be a
  **broadcast name/stats graphic overlay** (a translucent ghosted
  portrait animating over the live footage) fooling MediaPipe's landmark
  tracking — not real skater motion. This is now the single most
  important lesson from Phase 4a: **never trust an algorithmic
  candidate without opening the actual frame image first.**

---

## Planned Work Items (Remaining)

**4a**
- [ ] Resolve the torso-lean sign-flip: confirm same-skater identity across REST and EARLY-ACCELERATION segments
- [ ] Find and visually confirm a genuine settled-cruise window in `start_candidate_3.mp4`, or switch to a long-track clip if short track proves too short for one
- [ ] Once a valid cruise baseline exists, run the three-condition (unscaled/scaled/zscore-only) comparison

**4b**
- [ ] Visually audit the other 4 candidate videos (Ragne, Jorrit, Mia, Jan) the same way as Silovs
- [ ] Consider hand-trimming a clean segment around a confirmed genuine multi-person moment for a fuller test
- [ ] Only pursue new footage sourcing if no existing clip has a sufficiently long clean segment

**Explicitly out of scope for Phase 4:** aero/drag modeling (Phase 5), longitudinal tracking (Phase 6), general robustness to arbitrary compilation footage (documented recurring limitation, not being re-solved here).

---

## Honest Notes
- The 9/18 session is the clearest demonstration yet of this project's core methodology: an automated approach failed cleanly and informatively, a manual fallback found real data, and a plausible-looking algorithmic result was caught and corrected by visual verification rather than accepted at face value.
- The broadcast-content limitation is now confirmed across four separate videos across two different failure modes (cuts/title cards AND animated graphic overlays) — this is a general, structural property of broadcast/YouTube speed skating footage, not a series of isolated incidents, and should be stated as such in any final write-up.