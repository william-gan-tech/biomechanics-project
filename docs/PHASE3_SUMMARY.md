# Phase 3 Summary & Conclusion

## The Question
To what extent can relative bone-length scaling and proportional joint
coordinate normalization improve cross-subject generalization in deep
learning autoencoders to accurately detect neuromuscular fatigue across
diverse athletes with distinct stylistic variances?

## What Was Actually Tested
Split into two experiments for precision, since they test different
things:

- **Phase 3a** — Leave-One-Skater-Out ablation across 7 real, distinct
  athletes, comparing general motion-reconstruction loss variance under
  three conditions: unscaled (per-frame torso normalization), bone-length
  scaled (single calibrated scale per video), and z-score-only (image-size
  normalization without bone geometry). Standardization computed from
  training-pool data only, applied unchanged to the held-out skater —
  deliberately avoiding per-skater z-scoring, which would have erased the
  camera-distance differences scaling is meant to correct before the
  model ever saw them.
- **Phase 3b** — A second, more targeted experiment training only on other
  skaters' early-session ("fresh" proxy) data, then measuring the
  reconstruction-loss gap on a held-out skater's late-session ("fatigued"
  proxy) data. This is the experiment that actually matches the
  fatigue-detection wording of the question; 3a alone never distinguishes
  fresh from fatigued motion.

## The Result
With all 7 skaters, bone-length scaling showed **3–10x higher**
cross-subject variance than the unscaled baseline across every comparison
in both 3a and 3b — the opposite of the original hypothesis.

An outlier sensitivity check found this was driven almost entirely by one
athlete (Mia Manganello Kilburg) whose footage contains a mid-clip
broadcast camera cutaway, previously diagnosed and partially mitigated
with a frame-plausibility filter. **Excluding that one athlete (n=6), the
result reverses**: scaling shows lower variance than unscaled in 3 of 4
key comparisons.

Paired Wilcoxon signed-rank tests on every comparison, in both n=7 and n=6
configurations, came back statistically non-significant (p > 0.3
throughout).

## Why This Result Is Trustworthy
- **Reproduced twice** with consistent results before any correction was
  applied.
- **Two real calibration bugs were diagnosed and fixed** during testing
  (one-leg occlusion miscalibration, mid-clip cutaway) rather than
  dismissed as noise — and the result was recorded honestly even after
  both fixes didn't change the overall direction.
- **The outlier sensitivity check was run and reported in both
  directions** (n=7 and n=6) rather than only reporting whichever version
  looked better.
- **Proper paired statistical testing was applied**, and non-significance
  was reported plainly rather than reframed as a positive or null result.
- **A separate, unrelated claim of "100% variance reduction"** — found
  circulating in earlier project documentation — was investigated and
  retracted after discovering the script that produced it tested
  arbitrary time-chunks of a single video rather than distinct athletes,
  and contained a bug that would prevent it from ever completing a clean
  run. That retraction is a direct part of why this current result can be
  trusted: it replaced a fabricated-adjacent positive finding with a real,
  messier, honestly-reported one.

## Honest Limitations
- n = 6–7 is small; a single skater's footage quality visibly dominates
  the aggregate result.
- Fresh/fatigued labels in 3b are a position-in-session proxy, not
  verified ground truth (same limitation as Phase 1, applied here across
  subjects).
- Training was capped at 8 epochs / 1,500 windows per skater for runtime
  reasons — not yet tested whether more thorough training changes the
  result.
- Only one outlier mechanism has been identified and characterized; it's
  possible other skaters have subtler, undetected versions of the same
  camera-consistency problem.

## What It Actually Means
Bone-length scaling, as implemented (a single fixed calibration per
video), may genuinely help cross-subject generalization on clean,
single-camera-angle footage — but its aggregate benefit is fragile enough
to be reversed by one video with real-world camera inconsistencies.
**Footage-quality robustness may matter as much as, or more than, the
normalization method itself.** This reframes the practical bottleneck for
anyone deploying this kind of system: the harder problem may not be "does
bone-scaling work," but "how do you guarantee clean enough input video
for it to work."

## What This Set Up for Phase 4
The multi-person identity tracking system built 9/15 to harden the
calibration pipeline was directly motivated by this footage-robustness
finding. Phase 4 stress-tests that same system under a harder, more
realistic condition — genuinely simultaneous skaters at a start line,
rather than the sequential video-editing artifact that exposed the
original tracking weakness.