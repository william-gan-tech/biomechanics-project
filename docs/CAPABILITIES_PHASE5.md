# Phase 5: Form Analysis & Coaching Reference System

## Formal Research Question

> To what extent can bone-length-scaled joint-angle trajectories, compared
> against an explicitly-defined elite reference rather than a pooled
> average, be used to characterize and communicate specific technique
> deviations (stride mechanics, sit height, corner/cross asymmetry, lean
> consistency, arm swing, push direction) in both ice and inline speed
> skaters — and does incorporating fatigue-linked degradation improve the
> practical usefulness of that feedback?

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
9/21 torso-lean sign-convention artifact). This is a permanent, stated
limitation of the approach unless multi-camera triangulation is pursued
as separate future work, not a temporary gap to quietly fix later.

---

## Tier 1 — Core Technique Elements (this phase's actual scope)

### 5a — Elite-Anchor Reference Methodology
Stop comparing against a pooled average; explicitly designate reference
("target") skaters and formalize how deviation from them is measured and
reported. Foundation for every comparison tool below.

### 5b — Arm Swing
Shoulder-elbow-wrist angle, swing amplitude and rhythm. Not currently
measured anywhere in the pipeline, despite the landmarks already being
tracked by MediaPipe.

### 5c — Sit Height / Knee-Bend Depth
One of the most heavily-coached elements in speed skating ("get lower")
and currently entirely unmeasured — the pipeline tracks knee *angle* but
not how low the hip sits relative to the skater's own standing height.
Needs a per-skater standing-height calibration frame (similar pattern to
existing bone-length calibration) to normalize, since raw pixel height is
meaningless without it.

### 5d — Stride Rhythm & Cadence Consistency
How evenly-spaced/rhythmic strides are, distinct from single-frame
velocity/lean features. Built from existing peak-detection logic already
used elsewhere in the pipeline.

### 5e — Bilateral Asymmetry Validation
Directly test whether corners show genuinely more left-right asymmetry
than straightaways, using the `hip_lateral_asymmetry` metric built 9/22,
now with a real elite-anchor comparison (5a) rather than only descriptive
pooled stats.

### 5f — Fatigue-Linked Form Degradation
Reconnect the existing Phase 1/3 fatigue autoencoder to Tier 1 features,
segmented by technique phase and by early-vs-late session, so output
becomes "your corner lean consistency degrades by X as you fatigue"
instead of one blended whole-session number. Needs full-race footage
(not short clips) per skater — a real, specific data gap distinct from
current labeled segments.

### 5g — Unified Ice Form Report
Ties 5a-5f into one tool: upload a race clip, get a full form + fatigue
report across start/corner/straightaway, arm swing, sit height, and
asymmetry. The actual product-level deliverable for the ice-skating side
of the goal.

---

## Tier 2 — Real Refinements, Second Priority

### 5h — Push-Off Direction (not just magnitude)
Knee angle currently measures how bent the leg is, not which direction
the push is aimed (lateral vs. more backward) — a real, distinct
efficiency marker separate from bend depth.

### 5i — Push Phase vs. Glide Phase Separation
A stride currently gets treated as one undifferentiated blob. Separating
active push from passive glide/recovery would enable measuring
push-to-glide time ratio, a real efficiency indicator.

### 5j — Vertical Oscillation ("Bobbing")
Wasted up-down motion instead of forward propulsion. Genuinely hard to
measure reliably from a single, possibly-panning 2D camera — flagged as
harder than other Tier 2 items, may need a stability precondition (fixed
camera) to be trustworthy at all.

### 5k — Arm Swing Mode Transitions
Elite skaters typically shift from double-arm swing (starts) to
single-arm swing (cruising, one arm held behind the back). 5b measures
swing angle generically; this specifically detects the mode switch
itself — a distinct coaching point from swing amplitude alone.

### 5l — Finish/Sprint Technique
End-of-race form is often coached distinctly from mid-race cruising
(different urgency, different stride length). Not currently a separate
category anywhere in the phase plan.

### 5m — Full Sensitivity/Bug Audit of All New Metrics
Given that 9/21–9/22 found two real bugs (running-average error,
sign-convention artifact) purely through targeted leave-one-out
sensitivity testing, every Tier 1 and Tier 2 metric gets the same
scrutiny before being trusted for coaching use — not assumed correct
just because it computed without erroring.

---

## Tier 3 — Inline Skating (New Discipline, Not a Variant)

**Ice and inline are biomechanically distinct disciplines. Ice technique
references cannot be reused for inline, and vice versa — separate
reference datasets are required throughout, sharing only underlying code
and methodology, never pooled data.**

### 5n — Inline Groundwork: Footage & Detection
Zero inline footage currently exists in the dataset — every current clip
is ice speed skating. Before measuring double-push *quality*, first
confirm it's detectable at all (presence/absence, not quality yet).
Requires sourcing real inline footage from scratch.

### 5o — Ankle/Foot-Angle Feature (new landmark category)
Double-push relies on ankle/foot rotation for a secondary propulsion
phase that has no ice-skating equivalent (blade mechanics don't allow
it). Nothing in the current pipeline measures ankle angle at all — this
is new ground, not an extension of existing knee/hip features.

### 5p — Double-Push Detection & Quality Comparison
Once 5o's feature exists and 5n confirms real footage shows a detectable
signal, build the actual quality/form comparison tool — same pattern as
5g, but needs its own inline-specific elite anchor (a real, separate
reference-skater dataset, built the same multi-skater way ice data was
in Phase 4, starting again from n=1).

### 5q — Inline Corner/Crossover Mechanics
Inline skates use a different frame geometry (rocker/flat wheel
configuration) than ice blades — "correct" corner lean and crossover
technique is not simply "the same as ice, on wheels." This needs its own
dedicated reference profile, not a reuse of 5e's ice-corner findings.

### 5r — Inline Reference Dataset Growth
Same n-growth process ice skaters went through in Phase 4 (single
skater → verified multi-skater reference), applied to inline athletes
from scratch. An ongoing, separate data-collection track from the ice
side of the project.

---

## Sequencing Guidance

- **Tier 1 (5a–5g)** is buildable almost entirely from data already on
  hand — the fast, high-leverage track, and the actual minimum bar for a
  usable v1.
- **Tier 2 (5h–5m)** strengthens Tier 1 but isn't blocking — reasonable to
  interleave with Tier 1 or defer until after 5g's unified tool exists.
- **Tier 3 (5n–5r)** is a genuinely separate, longer-timeline project
  track requiring new footage and a from-scratch reference dataset — do
  not treat as a quick extension of the ice-skating work.