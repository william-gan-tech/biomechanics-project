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
  on (Lee Sang-Hwa's video)? This is a meaningfully different and harder
  case worth testing directly rather than assuming success.

---

## 🟡 Status: Not Yet Started

### Prerequisite: Footage Audit
Before any code work, confirm what start-line footage actually exists.
None of the current 7 ablation skaters' videos are confirmed to contain a
clean start sequence — most appear to be mid-race or full-session clips.
```powershell
# Check existing footage for start-line content before assuming it's there
Get-ChildItem .\data\*.mp4, .\videos\*.mp4
```
If no usable start footage exists yet, this needs to be sourced (same
`download_cross_skater_videos.py`-style approach used for Phase 3) before
Phase 4a can begin.

### Planned Work Items

**4a — New biomechanical features**
- [ ] Compute crouch/torso-lean angle (currently NOT in the 6-feature set — needs new geometry: angle between shoulder-hip vector and vertical)
- [ ] Compute initial acceleration profile (frame-to-frame velocity of hip centroid over the first N frames post-start)
- [ ] Define an objective "start phase" vs. "cruising phase" boundary per clip (proxy-based, same honesty standard as the Phase 3b fresh/fatigued split — document the assumption explicitly, don't claim it's ground truth)
- [ ] Compare bone-scaled vs. unscaled vs. z-score-only for these new features, reusing the three-condition ablation pattern from Phase 3a/3b

**4b — Multi-person tracking under real simultaneity**
- [ ] Source or identify footage with 2+ skaters genuinely simultaneous at a start line (not sequential clips)
- [ ] Run `diagnose_tracking_swap.py` against it as a first check, same as done for Lee Sang-Hwa's video, before assuming the tracker works
- [ ] If it fails, diagnose whether it's the same "no continuity across a discontinuity" issue or a new failure mode (e.g. genuinely identical-looking skaters in matching team uniforms, where appearance histograms also can't help)
- [ ] Document result honestly either way — a "does not generalize to true simultaneity, only to sequential passing" finding would itself be a legitimate, useful result

**Explicitly out of scope for Phase 4** (avoid scope creep into Phase 5/6 territory):
- Aerodynamic/drag modeling (that's Phase 5)
- Longitudinal/multi-session tracking (that's Phase 6)
- Full general robustness to arbitrary broadcast compilation footage (documented Phase 3 limitation, not being re-opened here)

---

## Honest Notes
- This phase directly builds on and stress-tests real Phase 3 infrastructure (tracking system, three-condition ablation methodology, statistical analysis pipeline) rather than starting fresh — reuse `run_bone_scaling_ablation.py`'s structure where possible instead of rewriting.
- No results yet. This file will be updated with real findings once Phase 4a/4b work begins — do not add placeholder numbers to this section ahead of actually running anything.