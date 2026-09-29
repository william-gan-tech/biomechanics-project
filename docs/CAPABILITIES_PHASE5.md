# Phase 5: Form Analysis & Coaching Reference System

## Formal Research Question

> To what extent can bone-length-scaled joint-angle trajectories, compared
> against an explicitly-defined elite reference rather than a pooled
> average, be used to characterize and communicate specific technique
> deviations (stride mechanics, sit height, corner/cross asymmetry, lean
> consistency, arm swing, push direction) in both ice and inline speed
> skaters — and does incorporating fatigue-linked degradation improve the
> practical usefulness of that feedback?

## Status (updated 9/28): 5a Complete & Re-Validated, 5b Built with a Documented Limitation, 5c Complete (9/24 limitation mostly resolved), 5d Built with Stated Limits, 5e Complete (Preliminary), 5f In Progress (First Real Result), 5g Not Started

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
9/21 torso-lean sign-convention artifact). Broadcast footage adds a second,
related problem: the camera zooms, pans and cuts, so anything measured in
body-scale *distance* units depends on how the scale is set (see the 9/28
per-video calibration bug under 5a/5c), and 2D angles change with camera
viewpoint. This is a permanent, stated limitation of the approach unless
multi-camera triangulation is pursued as separate future work.

---

## Tier 1 — Core Technique Elements (this phase's actual scope)

### 5a — Elite-Anchor Reference Methodology ✅ COMPLETE (9/23), RE-VALIDATED (9/28)
Stop comparing against a pooled average; explicitly designate reference
("target") skaters and formalize how deviation from them is measured and
reported. Foundation for every comparison tool below.

**Built:** `compare_to_elite_reference.py` — formalizes the labeled
dataset into an explicit `elite_reference_profile.csv` (mean + std per
phase + metric, using the corrected true-mean methodology from 9/21) and
a comparison tool for any new clip.

**Original check (9/23), superseded:** a Sven Kramer self-comparison
produced small z-scores (~±0.5-0.6). This turned out to be a weak test: a
skater already inside a 3-skater reference cannot score beyond about ±1.4,
so small scores were near-guaranteed.

**Proper validation (9/28):** `validate_elite_reference_loo.py` —
leave-one-out: each skater is compared against a reference rebuilt
*without* them, the same situation a genuinely new skater is in.
- The math is calibrated: 18.1% / 11.1% of held-out scores exceed the 80% /
  90% cutoffs (~20% / ~10% expected).
- But at n=3 per phase, a score must exceed about ±6 before it counts as
  unusual, and reading plain z-scores as normal would flag a third of the
  elite skaters as outliers. **Growing n is the main limit on this tool.**

**Corrections made 9/28:**
- Reference spread now uses sample std (`ddof=1`); population std understated
  spread at n=3.
- The tool now reports a prediction-adjusted score `t_pred = z / sqrt(1 + 1/n)`
  with a Student-t p-value, and only flags a metric as unusual at p < 0.10.
- **Per-video scale calibration bug (fixed):** every position in a video was
  divided by one fixed scale set from early calibration frames. For
  `start_candidate_3` that scale was 374 px against a measured torso of
  ~14-193 px (a broadcast close-up at the start), shrinking every distance
  2-26x; Silovs' scale drifted ~2x with camera zoom. All distance metrics are
  now rescaled per segment by the segment's 90th-percentile torso length
  (`SEGMENT_RESCALE` in `compare_to_elite_reference.py`). Diagnosed with
  `diagnose_sit_height_scale.py`. `start_candidate_3`'s hip-to-ankle scores
  moved from -4 to -9 to within ±1.2 in every moving phase.
- Velocity and acceleration are now per second, not per frame (the videos
  mix 25 and 29.97 fps).
- Ragne Wiklund is held out as `external_test`, so she stays a genuine outside
  test subject rather than part of the reference.

The profile now covers 12 metrics (6 original, 2 new sit-height metrics,
4 asymmetry metrics from 5e).

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

### 5c — Sit Height / Knee-Bend Depth ✅ COMPLETE (9/24), 9/24 LIMITATION MOSTLY RESOLVED (9/28)
One of the most heavily-coached elements in speed skating ("get lower")
and previously entirely unmeasured — the pipeline tracked knee *angle*
but not how low the hip sits relative to the ankle.

**Built:** ankle landmark extraction added to `preprocess_video.py`
(previously extracted for knee-angle math but discarded); computed
`hip_to_ankle_vertical_right` (bone-scale-normalized vertical hip-to-ankle
distance) and a standing-reference-calibrated `sit_height_ratio` in
`start_phase_features.py`. **Added 9/28:** `hip_to_ankle_2d_right` (full
2D hip-to-ankle distance) and `hip_to_ankle_lateral_right` (its horizontal
component), so crouch depth and lateral leg extension can be told apart.

**Validated:** tested against `start_candidate_3`'s verified `start_rest`
segment as the standing-reference calibration — REST correctly centered
near 1.0 (self-calibrated baseline); EARLY-ACCELERATION correctly showed
deeper crouching (mean 0.85, 25th percentile 0.66) — a physically
sensible, directionally-correct first result.

**Integrated:** added to `compare_to_elite_reference.py`'s metric set.

**9/24 limitation, revised 9/28:** the Ragne Wiklund external comparison
showed a near-fully-extended knee (174.6°) alongside the *shallowest*
hip-to-ankle distance of the session. On 9/24 this was attributed to the
metric measuring only the vertical component while push-off extends the leg
laterally. **On 9/28 this was found to be mostly the per-video scale bug
(see 5a):** Ragne's fixed scale was 7.2x too large, shrinking every distance
in her video. With per-segment rescaling, her hip-to-ankle distance is
*larger* than the reference (1.51 vs 1.07), which is consistent with a
straight knee — the contradiction is gone. A modest lateral component
remains (0.92 vs 0.41, p = 0.13), so the lateral explanation contributes,
but it was not the main cause.

**Remaining limitation:** start-rest segments where the torso is heavily
foreshortened (e.g. `start_candidate_3`, torso only 14-20 px) still give
wrong distances even with per-segment rescaling. Documented, not fixed.

### 5d — Stride Rhythm & Cadence Consistency 🟡 BUILT, WITH STATED LIMITS (9/28)
How evenly-spaced/rhythmic strides are, distinct from single-frame
velocity/lean features.

**Built:** `stride_rhythm.py` — stride period, cadence, rhythm evenness
(coefficient of variation of stride intervals) and left/right leg timing,
with a plot of every detection saved to `stride_rhythm_checks/` for visual
checking.

**Fixed through visual checking:** stride events are knee-flexion troughs
(extension peaks sit on flat plateaus and jitter); intervals spanning a
tracking gap are excluded; minimum stride spacing is set in seconds, not
frames.

**Sensitivity check instead of tuning:** no single detection threshold
worked for every clip, so rather than tune by eye on 8 segments, each
segment is run at three thresholds and only trusted if the result is
stable (5 of 9 segments).

**Results:** start-acceleration strides were very even for both measurable
skaters (rhythm CV 0.06, legs alternating near 0.5). `start_candidate_3`'s
straightaways were less even (0.14-0.35).

**Limits:** segments need 2-3+ seconds (4+ strides), so most short corner
segments can't be measured. Sven Kramer's 640x360 clip shows no measurable
stride pattern. Not yet added to the elite reference profile.

### 5e — Bilateral Asymmetry Validation ✅ COMPLETE, PRELIMINARY (9/28)
Directly test whether corners show genuinely more left-right asymmetry
than straightaways.

**Built:** `validate_asymmetry_by_phase.py`, plus three new asymmetry
measures that are less camera-dependent than `hip_lateral_asymmetry`
(which is really projected hip width, and changes as a skater rotates
relative to the camera): knee-angle difference, pelvic tilt (hip height
difference), and left/right trunk-lean difference. All four are in the
elite reference profile.

**Results (only 2 skaters have both phases):**
- **Pelvic tilt was higher in corners for both skaters** (~+25%), and
  between-skater averages agree (0.095 vs 0.058). The one consistent signal.
- **Knee asymmetry was mixed** (Patrick Meek 1.8x, `start_candidate_3` not).
  **The 9/19 knee finding (14° vs 2°) did not reproduce** on
  identity-verified segments.
- Projected hip width was *lower* in corners — likely body rotation relative
  to the camera, not technique.

### 5f — Fatigue-Linked Form Degradation 🟡 IN PROGRESS — FIRST REAL RESULT (9/28)
Compare each skater's form in early vs late laps of the same race, within
the same technique phase, and test whether the fatigue autoencoder picks up
the same change.

**Built:**
- `scan_full_race.py` — finds long, single-skater stretches in a full race.
- `make_frame_sheet.py` — frame grids for checking segments by eye.
- `validate_fatigue_form_degradation.py` — (A) Tier 1 metrics early vs late,
  (B) autoencoder reconstruction loss early vs late, (C) a check for whether
  that loss moves with camera zoom.
- `train_fatigue_model_v2.py` — the retrained autoencoder (see below).

**Data:** Inzell 2026 men's 5000m, Sander Eitrem (world record 5:58.52):
2 early corners (laps 2, 4) and 5 late corners (laps 10, 12). All scan
candidates contained a camera cut or phase change the scan missed, so every
segment was trimmed and phase-checked by eye.

**Result A — form metrics (1 skater, preliminary):** corner form largely
held up (torso lean -3.6%, sit height +3%, knee asymmetry +6%). One moderate
change: left/right trunk-lean difference rose from 2.2° to 8.2° — a candidate
fatigue marker. Stride period lengthened slightly (0.51 → 0.57 s). On-screen
hip velocity (+75%) is not usable on tracking-camera footage — actual lap
times got *faster*.

**Result B — fatigue autoencoder:** the original model was found to be
collapsed (see the running log). The retrained v2 model passes its collapse
checks. Early corners scored 0.35 and late corners 1.29, with no link to
camera zoom (ρ = 0.008). **But a control changed the conclusion:** with the
baseline set from lap 2 only, lap 4 (early, unseen) scored 2.81 — higher than
lap 10 (2.09). The apparent early-vs-late gap was mostly a baseline artifact,
so **there is no reliable fatigue signal from the model yet**. Lap 12 scored
far higher (~19), which could be real final-lap change or a camera-viewpoint
effect; one 2-second baseline segment is too little to tell.

**Next:** more early-race data (a longer baseline), a second skater, and
comparing only matching camera views.

### 5g — Unified Ice Form Report
Ties 5a-5f into one tool: upload a race clip, get a full form + fatigue
report across start/corner/straightaway, arm swing, sit height, and
asymmetry. The actual product-level deliverable for the ice-skating side
of the goal. Not yet started; depends on 5f.

---

## Tier 2 — Real Refinements, Second Priority

### 5h — Push-Off Direction (not just magnitude)
Knee angle currently measures how bent the leg is, not which direction
the push is aimed. The 9/24 motivation (sit height misreading lateral
extension) turned out to be mostly the scale bug, but a modest lateral
component remained, and the 9/28 `hip_to_ankle_lateral_right` metric is a
first step toward measuring push direction.

### 5i — Push Phase vs. Glide Phase Separation
A stride currently gets treated as one undifferentiated blob. Separating
active push from passive glide/recovery would enable measuring
push-to-glide time ratio, a real efficiency indicator. 5d's stride
detection could provide the stride boundaries.

### 5j — Vertical Oscillation ("Bobbing")
Wasted up-down motion instead of forward propulsion. Genuinely hard to
measure reliably from a single, possibly-panning 2D camera.

### 5k — Arm Swing Mode Transitions
Elite skaters typically shift from double-arm swing (starts) to
single-arm swing (cruising). 5b measures swing angle generically; this
specifically detects the mode switch itself.

### 5l — Finish/Sprint Technique
End-of-race form is often coached distinctly from mid-race cruising.
Not currently a separate category anywhere in the phase plan. 5f's lap-12
result is relevant here.

### 5m — Full Sensitivity/Bug Audit of All New Metrics
Given that 9/21–9/28 found repeated real bugs/limitations purely through
targeted sensitivity testing and honest investigation (running-average
error, sign-convention artifact, arm-swing landmark misdetection, the
identity-verification sequential-read bug, the per-video scale calibration
error, per-frame velocity units across mixed frame rates, and the collapsed
fatigue autoencoder), every Tier 1 and Tier 2 metric gets the same scrutiny
before being trusted for coaching use — not assumed correct just because it
computed without erroring. Partially done 9/28 (leave-one-out validation,
scale diagnostic, frame-rate fix, stride-threshold sensitivity check,
autoencoder collapse checks).

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
| 9/24 | Sit-height metric measures vertical-only hip-to-ankle distance, conflating genuine crouch depth with lateral leg extension during active push-off | **Mostly resolved 9/28** — the main cause was the scale bug below; 2D and lateral metrics added |
| 9/28 | Sven Kramer self-comparison was a weak validation (a reference member can't score beyond ~±1.4 at n=3) | **Replaced** by leave-one-out validation |
| 9/28 | Per-video scale calibration used early frames (often a broadcast close-up): 2-26x too large for `start_candidate_3`, 7.2x for Ragne Wiklund, ~2x drift within Silovs' video | **Fixed** (per-segment rescaling); start-rest with heavy foreshortening still unresolved |
| 9/28 | Velocity/acceleration were per frame, but videos mix 25 and 29.97 fps | **Fixed** (per-second units) |
| 9/28 | `labeled_segments.csv` rows silently merged when the file lacked a trailing newline (Patrick Meek 9/21, Eitrem 9/28) | **Fixed** (`--add` now guards against it; `--remove` added) |
| 9/28 | Histogram identity check unreliable in panning rail-cam shots (0.16-0.57 on one clearly identifiable skater — the crop is mostly changing background) | Documented; identity confirmed visually for those shots |
| 9/28 | Full-race scan runs can contain camera cuts and phase changes; automatic cut detection attempted and not adopted (whole-frame color stayed ~0.99 across real cuts) | Documented; every segment checked by eye with `make_frame_sheet.py` |
| 9/28 | Side rail-cam shots that track the curve look like straightaways (boards appear straight) but are corners | Documented; phase calls made by eye from frame sheets |
| 9/28 | Saved fatigue autoencoder (`skating_degradation_model.pth`) collapsed: identical output for any input. Trained on raw unstandardized features, while `pipeline_engine.py` feeds standardized inputs. Affects `app.py`'s fatigue timeline | **Retrained** as `skating_fatigue_model_v2.pth` (passes collapse checks); `app.py` not yet switched |
| 9/28 | Hip velocity measured on screen doesn't reflect real speed when the camera tracks the skater (Eitrem +75% while lap times got faster) | Documented; don't use hip velocity on tracking-camera footage |

---

## Data Sources

**Processed 9/28:**
- Inzell 2026 men's 5000m (Sander Eitrem world record vs Metoděj Jílek):
  https://www.youtube.com/watch?v=y8i0ln8xLj4 — 7 Eitrem corner segments
  and 1 straightaway, lap-labeled for 5f.

**Found, not yet processed:**
- Corner-technique video identified 9/22 as a candidate for expanding the
  corner dataset: https://www.youtube.com/watch?v=C8lYMjOxWEI
- Beijing 2022 men's 5000m full replay (several pairs, candidate second
  skater for 5f): https://www.youtube.com/watch?v=ulvKaqIqK6Y
- Milano Cortina 2026 men's 5000m medal performances:
  https://www.youtube.com/watch?v=z92Xza9kMXQ

---

## Sequencing Guidance

- **Tier 1 (5a–5g)**: 5a re-validated, 5b built with a known limitation, 5c
  done with its main limitation resolved, 5d built with stated limits, 5e
  done (preliminary). **5f is the current focus**: it needs more early-race
  data and a second skater. 5g depends on 5f.
- **Biggest single constraint across Tier 1: sample size.** At n=3 per phase,
  the comparison tool can't reliably flag anyone; 5e rests on 2 skaters and
  5f on 1.
- **Tier 2 (5h–5m)**: strengthens Tier 1 but isn't blocking. 5m is partly
  done. Switching `app.py` to the v2 fatigue model belongs here too.
- **Tier 3 (5n–5r)**: a genuinely separate, longer-timeline project track
  requiring new footage and a from-scratch reference dataset.
