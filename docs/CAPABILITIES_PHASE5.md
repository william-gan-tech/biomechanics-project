# Phase 5: Form Analysis & Coaching Reference System

## Formal Research Question

> To what extent can bone-length-scaled joint-angle trajectories, compared
> against an explicitly-defined elite reference rather than a pooled
> average, be used to characterize and communicate specific technique
> deviations (stride mechanics, sit height, corner/cross asymmetry, lean
> consistency, arm swing, push direction) in both ice and inline speed
> skaters — and does incorporating fatigue-linked degradation improve the
> practical usefulness of that feedback?

## Status (updated 9/24): 5a Complete & Verified, 5b Built with a Documented Limitation, 5c Complete, Validated & Integrated

## What "Complete" Means Here — Stated Honestly

No form-analysis model is ever fully complete. This phase targets a
defensible v1: the technique elements a coach would look at first,
covering ice skating comprehensively before extending to inline (a
genuinely distinct discipline, not a variant). Deeper refinements are
listed explicitly as Tier 2/3 rather than silently omitted.

**Structural limitation, stated upfront:** every feature in this pipeline
is derived from a single 2D camera view, not true 3D measurement. Lean
angle, knee-bend depth, and push direction are inherently 3D quantities
being approximated from 2D video — this already caused one real bug (the
9/21 torso-lean sign-convention artifact) and a second, distinct issue
found 9/24 (the sit-height metric's vertical-only measurement conflating
crouch depth with lateral leg extension — see 5c below). This is a
permanent, stated limitation of the approach unless multi-camera
triangulation is pursued as separate future work.

---

## Tier 1 — Core Technique Elements (this phase's actual scope)

### 5a — Elite-Anchor Reference Methodology ✅ COMPLETE (9/23)
Stop comparing against a pooled average; explicitly designate reference
("target") skaters and formalize how deviation from them is measured and
reported. Foundation for every comparison tool below.

**Built:** `compare_to_elite_reference.py` — formalizes the labeled
dataset into an explicit `elite_reference_profile.csv` (mean + std per
phase + metric, using the corrected true-mean methodology from 9/21) and
a real z-score comparison tool for any new clip.

**Verified:** self-comparison sanity check — Sven Kramer against a
reference profile he's a member of produced small z-scores (~±0.5-0.6)
across all metrics, confirming the math works correctly.

### 5b — Arm Swing 🟡 BUILT, WITH A REAL UNRESOLVED LIMITATION (9/23)
Shoulder-elbow-wrist angle, swing amplitude and rhythm. Not previously
measured anywhere in the pipeline, despite the landmarks already being
tracked by MediaPipe.

**Built:** elbow/wrist landmark extraction added to `preprocess_video.py`
(indices 13-16); elbow-bend angle and frame-to-frame swing amplitude
computed in `start_phase_features.py`.

**Real problem found, not hidden:** direct visual inspection found elbow
angle swinging 150+ degrees frame-to-frame in early clip frames despite
the skater's visible arm position being nearly static across the same
~0.1s window — confirmed landmark misdetection, not real fast motion.

**Fix attempted and found insufficient:** added MediaPipe's own
per-landmark visibility/confidence scores, hypothesizing low confidence
would flag bad frames directly. Tested explicitly: known-bad frames
scored 0.76-0.81, not meaningfully different from the clip's normal
range (0.58-0.97) — MediaPipe is confidently wrong here, not uncertain,
a harder failure mode than a threshold filter can catch.

**Status:** documented as a genuine, unresolved limitation rather than
chased further. Practical mitigation: manual spot-checking of arm-swing
segments against visible video before trusting them, and skipping early
clip frames by default when building arm-swing reference data.

### 5c — Sit Height / Knee-Bend Depth ✅ COMPLETE, VALIDATED & INTEGRATED (9/24)
One of the most heavily-coached elements in speed skating ("get lower")
and previously entirely unmeasured — the pipeline tracked knee *angle*
but not how low the hip sits relative to the ankle.

**Built:** ankle landmark extraction added to `preprocess_video.py`
(previously extracted for knee-angle math but discarded); computed
`hip_to_ankle_vertical_right` (bone-scale-normalized vertical hip-to-ankle
distance) and a standing-reference-calibrated `sit_height_ratio` in
`start_phase_features.py`.

**Validated:** tested against `start_candidate_3`'s verified `start_rest`
segment as the standing-reference calibration — REST correctly centered
near 1.0 (self-calibrated baseline); EARLY-ACCELERATION correctly showed
deeper crouching (mean 0.85, 25th percentile 0.66) — a physically
sensible, directionally-correct first result.

**Integrated:** added to `compare_to_elite_reference.py`'s metric set;
elite reference profile rebuilt with 6 metrics across 3 skaters per
phase.

**Real limitation discovered during external testing (9/24):** a genuine
external comparison (Ragne Wiklund, not part of the reference profile)
produced a physically implausible result — near-fully-extended knee
(174.6°) alongside the *shallowest* hip-to-ankle vertical distance of
the whole session. Initially suspected tracking contamination (the
segment did turn out to have 60% two-skater overlap); after finding and
fixing a real bug in the identity-verification tool (see below) and
re-testing on a properly verified clean segment, the odd pattern
persisted even with identity confirmed correct. Root cause: the metric
measures only the *vertical* component of hip-to-ankle distance, but
speed-skating push-off extends the leg *laterally*, not straight down —
a fully extended leg pushed outward during an active push can show a
small vertical-only distance despite being completely straight.
**Documented as a genuine, unresolved conceptual limitation**: the
current metric cannot distinguish real knee-bend crouching from lateral
leg extension angle. A future fix would need true 2D (not vertical-only)
hip-ankle distance, or a separate lateral-extension metric measured
independently from vertical crouch depth.

### 5d — Stride Rhythm & Cadence Consistency
How evenly-spaced/rhythmic strides are, distinct from single-frame
velocity/lean features. Built from existing peak-detection logic already
used elsewhere in the pipeline. Not yet started.

### 5e — Bilateral Asymmetry Validation
Directly test whether corners show genuinely more left-right asymmetry
than straightaways, using the `hip_lateral_asymmetry` metric built 9/22,
now with a real elite-anchor comparison (5a, available) rather than only
descriptive pooled stats. Infrastructure exists; formal validation not
yet run.

### 5f — Fatigue-Linked Form Degradation
Reconnect the existing Phase 1/3 fatigue autoencoder to Tier 1 features,
segmented by technique phase and by early-vs-late session. Needs
full-race footage (not short clips) per skater — a real, specific data
gap distinct from current labeled segments. Not yet started.

### 5g — Unified Ice Form Report
Ties 5a-5f into one tool: upload a race clip, get a full form + fatigue
report across start/corner/straightaway, arm swing, sit height, and
asymmetry. The actual product-level deliverable for the ice-skating side
of the goal. Not yet started; depends on 5d-5f.

---

## Tier 2 — Real Refinements, Second Priority

### 5h — Push-Off Direction (not just magnitude)
Knee angle currently measures how bent the leg is, not which direction
the push is aimed. **Now additionally motivated by the 9/24 finding**:
push direction (lateral vs. vertical) is exactly the missing dimension
that caused 5c's sit-height metric to misread a lateral leg extension as
a shallow crouch — this Tier 2 item would directly help disambiguate
that.

### 5i — Push Phase vs. Glide Phase Separation
A stride currently gets treated as one undifferentiated blob. Separating
active push from passive glide/recovery would enable measuring
push-to-glide time ratio, a real efficiency indicator.

### 5j — Vertical Oscillation ("Bobbing")
Wasted up-down motion instead of forward propulsion. Genuinely hard to
measure reliably from a single, possibly-panning 2D camera.

### 5k — Arm Swing Mode Transitions
Elite skaters typically shift from double-arm swing (starts) to
single-arm swing (cruising). 5b measures swing angle generically; this
specifically detects the mode switch itself.

### 5l — Finish/Sprint Technique
End-of-race form is often coached distinctly from mid-race cruising.
Not currently a separate category anywhere in the phase plan.

### 5m — Full Sensitivity/Bug Audit of All New Metrics
Given that 9/21–9/24 found four real bugs/limitations purely through
targeted sensitivity testing and honest investigation (running-average
error, sign-convention artifact, arm-swing landmark misdetection,
sit-height vertical-only conflation, plus the 9/24 identity-verification
sequential-read bug), every Tier 1 and Tier 2 metric gets the same
scrutiny before being trusted for coaching use — not assumed correct
just because it computed without erroring.

---

## Tier 3 — Inline Skating (New Discipline, Not a Variant)

**Ice and inline are biomechanically distinct disciplines. Ice technique
references cannot be reused for inline, and vice versa — separate
reference datasets are required throughout, sharing only underlying code
and methodology, never pooled data.**

### 5n — Inline Groundwork: Footage & Detection
Zero inline footage currently exists in the dataset. Requires sourcing
real inline footage from scratch.

### 5o — Ankle/Foot-Angle Feature (new landmark category)
Double-push relies on ankle/foot rotation with no ice-skating equivalent.
Nothing in the current pipeline measures ankle *angle* (only ankle
*position*, added 9/24 for sit-height) — this is new ground.

### 5p — Double-Push Detection & Quality Comparison
Needs its own inline-specific elite anchor, built the same multi-skater
way ice data was, starting again from n=1.

### 5q — Inline Corner/Crossover Mechanics
Inline skates use a different frame geometry than ice blades — needs its
own dedicated reference profile, not a reuse of 5e's ice-corner findings.

### 5r — Inline Reference Dataset Growth
Same n-growth process ice skaters went through in Phase 4, applied to
inline athletes from scratch.

---

## Real Bugs & Limitations Found This Phase (Running Log)

| Date | Finding | Status |
|---|---|---|
| 9/23 | Arm-swing elbow angle can be confidently wrong (landmark misdetection), not just noisy; visibility-score fix attempted and found insufficient | Documented, unresolved |
| 9/24 | `check_same_skater_identity.py` reopened VideoCapture and seeked directly to isolated frames per call, breaking MediaPipe VIDEO mode's reliance on temporal continuity — caused false negative detections on frames independently confirmed to have a visible person | **Fixed and verified** (sequential-read rewrite) |
| 9/24 | Sit-height metric measures vertical-only hip-to-ankle distance, conflating genuine crouch depth with lateral leg extension during active push-off | Documented, unresolved |

---

## Pending Data Sources (found, not yet processed)

- Corner-technique video identified 9/22 as a candidate for expanding the
  corner dataset:
  https://www.youtube.com/watch?v=C8lYMjOxWEI

---

## Sequencing Guidance

- **Tier 1 (5a–5g)**: 5a done, 5b built with a known limitation, 5c done
  and integrated. 5d–5g remain — still the fast, high-leverage track.
- **Tier 2 (5h–5m)**: strengthens Tier 1 but isn't blocking. 5h now has
  concrete motivation from the 9/24 sit-height finding.
- **Tier 3 (5n–5r)**: a genuinely separate, longer-timeline project track
  requiring new footage and a from-scratch reference dataset.