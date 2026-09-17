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

## 🟢 Status: Started — Real Preliminary Results (9/16)

### Prerequisite Footage Audit — Complete
Built `audit_footage_for_starts.py` to check whether existing footage
already contains genuine simultaneous multi-person content, before
assuming new video needed to be sourced.

**Result:** 5 of 7 existing skater videos show meaningful simultaneous
multi-person content when sampled: Haralds Silovs (42.2%), Ragne Wiklund
(33.3%), Jorrit Bergsma (32.4%), Mia Manganello Kilburg (17.2%), Jan
Blokhuijsen (11.7%). **Conclusion: Phase 4b testing could begin
immediately on existing footage — new video sourcing was not an
immediate blocker, contrary to the original plan's assumption.**

### 4a — Feature Engineering: Built and Verified

Built `start_phase_features.py` computing three new features not present
in the Phase 3 feature set:
1. **Torso-lean/crouch angle** — angle between the shoulder-hip vector
   and vertical, in degrees.
2. **Hip velocity** — frame-to-frame hip-centroid displacement.
3. **Hip acceleration** — frame-to-frame change in velocity.

**Honest limitation stated in the code itself:** computed from
`norm_right_hip_x/y` and `norm_right_shoulder_x/y` only — the right side,
since that's what the existing pipeline saves. A real asymmetry given
skating is not bilaterally symmetric, especially at push-off. Left-side
or averaged features would need pipeline changes; not done here.

**Verified working** against real extracted data (Sven Kramer's
reference video, since the feature cache was empty at time of testing).
The math runs correctly and produces sensible-looking output. **No real
start-phase finding yet** — this test was run against non-start footage
specifically to validate the code, and correctly produced a pattern that
does NOT resemble a real start (velocity was higher in the "start" proxy
window than the rest of the clip, backwards from what a real start
should show) — a useful negative confirmation that the math isn't
spuriously finding patterns that aren't there.

### 4b — Tracking Test on Real Footage: First Genuine Success + New Limitation Found

Ran `diagnose_tracking_swap.py` (reused from Phase 3) against Haralds
Silovs' footage.

**Real success — first confirmed correct multi-person resolution on real
footage:** At frame 1786, two genuinely simultaneous people were detected
with ambiguous position distances (0.0354 vs. 0.0396 — closer together
than the ambiguity threshold). The appearance-histogram tiebreak
correctly activated and selected the higher-similarity candidate (0.959
vs. 0.944 correlation). This is the tracking system working exactly as
designed, on real footage, for the first time.

**New limitation found, correctly diagnosed:** A ~100-frame (3+ second)
stretch of zero detected candidates (frames 1792–1889) was visually
confirmed to be a title-card/text overlay, not a tracking failure.
Same underlying pattern as the previously-documented Lee Sang-Hwa
compilation-cut limitation (9/15) — broadcast/YouTube footage content
inconsistency, not the tracking algorithm, remains the dominant practical
bottleneck. This means the footage audit's "42.2% simultaneous"
statistic for Silovs likely includes some detection noise around
non-skating content rather than only sustained genuine two-skater
simultaneity — worth treating as an upper bound, not a precise measure.

---

## Planned Work Items (Remaining)

**4a**
- [ ] Test feature math against a real, visually-confirmed start-line moment (not yet done — tonight's test used non-start footage to validate the code only)
- [ ] Define the start-vs-cruising boundary explicitly per clip, once real start footage/segments are identified
- [ ] Run the three-condition (unscaled/scaled/zscore-only) comparison once real start data is available

**4b**
- [ ] Visually audit the other 4 candidate videos (Ragne, Jorrit, Mia, Jan) the same way as Silovs, to find a cleaner sustained-simultaneity segment if one exists
- [ ] Consider hand-trimming a clean segment around a confirmed genuine multi-person moment (like frame 1786) for a fuller test, same mitigation used for Lee Sang-Hwa
- [ ] Only pursue new footage sourcing if no existing clip has a sufficiently long clean segment

**Explicitly out of scope for Phase 4:** aero/drag modeling (Phase 5), longitudinal tracking (Phase 6), general robustness to arbitrary compilation footage (documented Phase 3/4b limitation, not being re-solved here).

---

## Honest Notes
- Both 4a and 4b made real, verified progress on the first working day, faster than the original plan assumed, mainly because the footage audit avoided an unnecessary footage-sourcing delay.
- The recurring "broadcast footage has title cards / cuts" limitation is now confirmed across three separate videos (Mia in Phase 3, Lee Sang-Hwa in the 9/15 tracking test, Silovs tonight) — this is a real, general pattern for this kind of footage, not a one-off, worth stating as a class of limitation rather than three separate incidents.