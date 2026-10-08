# Phase 5 Summary & Conclusion

> **DRAFT (10/7) — written by Claude for the researcher's review.** Not
> final until reviewed. Numbers are taken from the analysis outputs and
> `docs/CAPABILITIES_PHASE5.md`; please check the interpretation and wording.

## The Question
To what extent can bone-length-scaled joint-angle trajectories, compared
against an explicitly-defined elite reference rather than a pooled
average, be used to characterize and communicate specific technique
deviations (stride mechanics, sit height, corner/cross asymmetry, lean
consistency, arm swing, push direction) in ice speed skaters — and does
incorporating fatigue-linked degradation improve the practical usefulness
of that feedback?

(Phase 5's scope is Tier 1, items 5a-5g. Tier 2 refinements and Tier 3
inline skating are future work.)

## What Was Actually Tested
- **5a — Elite-anchor reference:** each skater compared against a
  reference built from the other elite skaters, validated leave-one-out.
- **5b-5e — Technique measures:** arm swing, sit height, stride rhythm,
  and left/right asymmetry by phase.
- **5f — Fatigue-linked form change:** the same skater's form in early vs
  late laps of a full race, within the same technique phase.
- **5g — Form report:** one report per skater combining all of the above.

**Data:** 163 lap-labeled corner/straightaway segments from full races
(188 labeled segments in total, including start and earlier reference
clips), from **9 elite skaters across 6 races** — the Inzell 2026 5000m world
record (Eitrem), the Beijing 2022 Olympic 5000m (Kramer, Roest, Bergsma,
Bloemen, Rijhnen), and four ISU World Cup single-race videos including
two 10000m races (Ghiotto's world record, van der Poel). Every segment's
phase (corner/straightaway) and **camera shot type** (side/front/wide) was
called by the researcher from frame sheets, after automated scanning
narrowed thousands of frames to candidate stretches.

## The Result

**5a — The elite reference works.** With 12 corner and 11 straightaway
reference skaters, leave-one-out calibration is on target: 20.7% / 10.1%
of held-out scores fall below p = 0.20 / 0.10 (20% / 10% expected). Start
phases remain thin (3 skaters).

**5c — Sit height:** knee angle is the dependable sit-height measure. The
torso-scaled hip-to-ankle distance exaggerates changes and is no longer
used to flag anything.

**5d — Stride rhythm:** measurable on 38 segments; noisy on broadcast
footage because continuous stretches are short.

**5e — Asymmetry by phase:** the preliminary finding (pelvic tilt higher in
corners, 2 of 2 skaters) did not replicate with 11 skaters. Controlled for
camera shot type, corners were *less* asymmetric than straightaways
(trunk-lean left/right difference lower for 7 of 8 skaters, p = 0.008) —
most likely because these 2D left/right measures capture the side-to-side
alternation of straightaway strokes rather than corner technique.

**5f — Fatigue-linked form change (the central question):**

| Late-race change, per skater (n = 9) | All shots | Within camera shot type |
|---|---|---|
| **Trunk-lean left/right difference** | **+3.8°, 8 of 9 skaters (p = 0.002)** | +1.7°, 6 of 9 (p = 0.11) |
| Knee angle | +3.9°, 8 of 9 (p = 0.14) | +2.8°, 4 of 9 (p = 0.47) |

- Trunk-lean asymmetry rising late in the race is the **most consistent
  change**, and it holds when uncertain segment boundaries are trimmed
  (8 of 9 at every trim level).
- But it becomes **small and unstable once camera shot type is
  controlled**, so it is a consistent lead rather than an established
  fatigue marker.
- **Knee straightening is not a general sign** — strong for some skaters
  (Bergsma, Kramer), absent or reversed for others.
- **The fatigue autoencoder does not detect late-race change.** The saved
  Phase 1 model was found collapsed (identical output for any input); a
  retrained version passed collapse checks but showed no reliable
  late-race signal against a fair early-race baseline.

**5g — Form report:** works from labeled segments for all skaters.
Automatic corner/straight detection was tested (77% segment accuracy vs a
66% baseline; target 85%) and is not yet good enough.

## Why This Result Is Trustworthy
- **Every comparison was checked against the camera.** Late-race laps were
  first thought to be filmed more frontally (which would explain the
  pattern away); researcher-checked shot labels showed the shot mix was
  almost the same early and late, and that the earlier impression came
  from skewed provisional labels and a camera measure that partly tracks
  posture. Both the plain and the camera-controlled results are reported.
- **Each skater counts once.** A per-skater analysis replaced segment-level
  statistics after they were found to over-count segments from the same
  skater (knee p = 0.005 at segment level vs p = 0.47 per skater within
  shot type).
- **Robustness checks were run, not assumed:** segment-boundary trimming,
  leave-one-out reference calibration, a fair early-race baseline for the
  autoencoder, and collapse checks on every model.
- **Real bugs were found and fixed through these checks** — including a
  2-26x camera-scale calibration error that had produced a false
  conclusion on 9/24, and the collapsed fatigue model behind earlier
  phases' app output.
- **Negative results are reported as results:** the 5e replication
  failure, the autoencoder's lack of signal, the classifier falling short,
  and a form-drift timeline that pointed the wrong way on whole clips.

## Honest Limitations
- **Single-camera 2D broadcast video is the dominant limitation.** Camera
  zoom, angle changes and shared frames caused most of this phase's hard
  problems, and the camera-controlled 5f result is weak partly because
  matching shots early and late leaves little data.
- 9 skaters, 6 races, mostly men's long-distance elite skaters.
- "Late in the race" is a proxy for fatigue, not a measurement of it.
- Several metrics were tested; p-values are not corrected for that.
- Phase and shot-type calls are one person's judgement; side vs front was
  sometimes genuinely ambiguous.
- Arm swing (5b) is unreliable because elbow landmarks can be confidently
  wrong.

## What It Actually Means
For the research question: **comparing a skater against an elite
reference with interpretable joint-angle measures works and is well
calibrated**, which supports the "characterize and communicate technique
deviations" half. For the fatigue half, **interpretable form measures were
more informative than the deep-learning autoencoder**, and they point to
trunk-lean asymmetry increasing late in races — but broadcast footage
cannot confirm it once the camera is controlled. So fatigue modeling does
**not yet** improve the practical usefulness of the feedback on this kind
of footage. The project's strength shifts from "AI fatigue detection"
toward measurable, explainable form analysis.

## What This Sets Up for Phase 6
Phase 5's limits point clearly at what would move the project forward:
- **Controlled footage:** a fixed side-on camera, filming skaters through a
  hard set of laps with a real fatigue measure (lap times, heart rate).
  This removes the camera problem and gives 5f a true fatigue reference —
  the most direct way to confirm or rule out the trunk-asymmetry lead, and
  likely to make automatic phase detection work.
- **Dual-skater tracking** with a proper multi-object tracker: roughly half
  of all broadcast candidate stretches were unusable because both skaters
  were in frame.
- Multi-camera 3D and inline skating remain longer-term options.
