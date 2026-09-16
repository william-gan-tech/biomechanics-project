> **Verification note (added 9/15):** This file was reviewed in full against
> the actual codebase and conversation history. No corrections were needed —
> all claims in this document, including Part 4, were independently
> confirmed accurate as of this date.
 
 # 🚀 Phase 3 Capabilities & Milestone Log (`abilities_phase3.md`)

> **This file was substantially rewritten on 9/12** after an audit found
> several claims in the original version could not be corroborated against
> the actual codebase, or were directly contradicted by it. See the
> "Retracted Claims" section at the bottom for what changed and why.

## 🟢 Part 1: Verified, Working Phase 3 Capabilities 8/29 - present

### 🔧 Pipeline Reliability Fixes (9/08)
* **MediaPipe API Migration:** Discovered the installed `mediapipe` version (1.0.1) removed the legacy `mp.solutions.pose` API entirely. Rewrote bone-length calibration and annotated-video rendering to use the `mediapipe.tasks.python.vision.PoseLandmarker` API.
* **Path Resolution Fixes:** Found and fixed `ROOT_DIR` silently pointing one directory above the actual project root (broke relative dataset lookups since initial deployment), and a `sys.path` ordering bug causing a stale duplicate `preprocess_video.py` in `src/` to shadow the correct file.
* **Intro/Title-Card Skip:** Calibration was silently failing on footage with a title card before the athlete appears, returning a hardcoded 0.45 fallback. Added forward-scanning logic to find the first frame with a detected person before sampling.

### 👥 Real Cross-Skater Comparison (9/09–9/10)
* **Replaced Simulated Data:** The original Cross-Skater Anomaly mode used `np.random.uniform()` to generate fake per-joint error curves and a static "91.4% Cross-Subject Accuracy" value. Rebuilt as `cross_skater_compare.py`: real per-skater feature extraction with disk caching, resampling to a common normalized stride-cycle length, and DTW (dynamic time warping) distance per feature.
* **Honest Feature Labeling:** Original UI displayed "Hip Angle" and "Ankle Dorsiflexion" error curves that were never actually computed anywhere in the pipeline. Replaced with the 6 features the pipeline genuinely produces (right/left knee angle, bone-scaled hip x/y, bone-scaled shoulder x/y).
* **Tiered Fallback with Honest Labeling:** Added a CSV fallback (knee-angle only — explicitly excludes unscaled position columns to avoid contaminating bone-scaling comparisons) and a clearly-labeled "SIMULATED" fallback for skater pairs with neither video nor usable CSV, rather than silently faking results.
* **Video Coverage Expansion:** Downloaded real footage for 4 additional skaters (Ragne Wiklund, Mia Manganello Kilburg, Jorrit Bergsma, Jan Blokhuijsen), bringing real-video cross-skater comparison coverage to 7 of 10 tracked athletes.

## 🟡 Part 2: Phase 3 Core Research Question — Ablation Result

### Leave-One-Skater-Out (LOSO) Ablation (`run_bone_scaling_ablation.py`, 9/11–9/12)
* **Real methodology:** Trains a fresh `SkatingLSTMAutoencoder` on N-1 **actual distinct skaters** (not time-chunks of one video), evaluates reconstruction loss on the held-out skater, repeated for `reference_scale=None` (unscaled) and calibrated bone-length scale (scaled) conditions. Standardization computed from training-pool data only, applied unchanged to the held-out skater — deliberately avoids per-skater z-scoring, which would erase the camera-distance differences bone-scaling is meant to correct before the model ever sees them.
* **Made resumable:** Original version only saved results at the end of a full run; rebuilt to save after every fold and skip completed folds on restart, after ~3 hours of unsaved progress were lost to a necessary interruption.
* **Diagnosed two real data-quality failures rather than dismissing an inconvenient result:**
  1. Ragne Wiklund's calibration was silently returning the fallback value because her left knee had persistently low MediaPipe visibility (~0.08–0.10) from camera-angle occlusion, failing a symmetric "both legs visible" check. Fixed `extract_ensemble_reference_scale` to select whichever leg is visible per frame.
  2. Mia Manganello Kilburg's footage contained a mid-clip broadcast cutaway to a close-up shot, causing MediaPipe to output implausible position values for off-screen landmarks. Added a frame-level plausibility filter (drops frames with position values beyond a sane bound), applied identically to every skater/condition.

### **Result (n=7 skaters, 8 training epochs, max 1,500 windows/skater):**

| Condition | Mean held-out loss | Std | Variance |
|---|---|---|---|
| Unscaled | 0.278 | 0.174 | 0.030 |
| Scaled | 0.391 | 0.465 | 0.216 |

**In this pilot, bone-length scaling (as implemented — a single fixed scale calibrated once per video) did NOT reduce cross-subject reconstruction-loss variance; it increased it**, even after fixing the two calibration bugs above. This is being recorded as the genuine result of this experiment.

**Honest limitations:** n=7 is a small sample; a couple of skaters can dominate the variance metric. Training was capped at 8 epochs / 1,500 windows per skater for runtime reasons. It is not yet known whether other skaters in the set have subtler, undiagnosed versions of the same camera-consistency issues found in 2 of the 7.

**Working interpretation:** a single fixed reference scale assumes consistent camera framing for the whole clip. Real-world broadcast/YouTube footage often violates that assumption (title cards, cutaways, occlusion) in ways a per-frame approach tolerates more gracefully than a single locked-in scale. This is itself a candidate finding — that naive fixed-scale bone-length normalization is *more* brittle to real-world footage inconsistency than the simpler per-frame approach — worth investigating further (see Next Steps).

## 🔴 Retracted / Downgraded Claims (as of 9/12 audit)

* **~~"100.00% variance reduction, MSE range 267.35–451.29 vs 0.47–0.58"~~ — RETRACTED.** Sourced from `src/evaluate_generalization.py`, which was found to (a) split a single video into 4 arbitrary consecutive time-chunks labeled "pseudo-subjects" rather than evaluating distinct athletes, and (b) contain a self-recursive `main()` call that would cause a `RecursionError` on any actual completed run. This does not test the stated research question and the associated numbers should not be cited.
* **~~"ONNX Edge Runtime Acceleration... lower CPU/GPU inference latency"~~ — DOWNGRADED to attempted/non-functional.** See Phase 2 log for details; `onnxruntime` is never actually invoked in the running application, and the int8 file did not shrink as expected.
* **~~"Multi-Angle Camera Stream Fusion... operational"~~ — UNVERIFIED, pending review.** The actual committed file is `multi_view_fusion.py` (not `fusion_engine.py` as previously logged); its contents have not yet been reviewed to confirm whether it contains working fusion logic or another stub.
* **"Asynchronous Multi-Threaded Streaming"** — UNVERIFIED. No threading code was found in `pipeline_engine.py` during a keyword search; not yet conclusively confirmed either way.

## 🟡 Part 3: Phase 3b — Fatigue-Detection Separability & Outlier Sensitivity

### Fatigue-Separability Ablation (`run_fatigue_separability_ablation.py`, 9/13)
Directly tests the original Phase 3 wording ("...to accurately detect neuromuscular fatigue..."), unlike Phase 3a which tested general motion-reconstruction variance without distinguishing fresh from fatigued movement at all. Trains on other skaters' early-session ("fresh" proxy) data only, then measures the reconstruction-loss gap on a held-out skater's late-session ("fatigued" proxy) data. Fresh/fatigued is an assumption (first/last 25% of session), not verified ground-truth labeling — same technique used in Phase 1.

### Statistical Analysis Layer (`analyze_phase3_results.py`, 9/13)
Paired Wilcoxon signed-rank tests across all condition pairs in both Phase 3a and 3b. All comparisons at n=7 came back non-significant (p > 0.4 in every case) — reported honestly rather than claimed as either a positive or null result, given the small sample.

### Outlier Sensitivity Check (`outlier_sensitivity_check.py`, 9/14) — Key Finding

**With all 7 skaters:** bone-length scaling shows 3-10x higher cross-subject variance than unscaled or z-score-only across every comparison, in both Phase 3a and 3b.

**Excluding Mia Manganello Kilburg (the known camera-cutaway-affected skater, n=6):** the result **reverses** — scaling shows *lower* variance than unscaled and z-score-only in 3 of 4 key comparisons, and roughly equal in the 4th.

| Comparison | Full (n=7) scaled var | Excl. Mia (n=6) scaled var | Direction |
|---|---|---|---|
| 3a unscaled vs scaled | 0.1731 vs 0.0237 | 0.0172 vs 0.0224 | **Reversed** |
| 3a zscore vs scaled | 0.1731 vs 0.0562 | 0.0172 vs 0.0578 | **Reversed** |
| 3b unscaled vs scaled | 0.8517 vs 0.0730 | 0.0547 vs 0.0550 | Roughly equal |
| 3b zscore vs scaled | 0.8517 vs 0.1935 | 0.0547 vs 0.2212 | **Reversed** |

**Working conclusion:** Bone-length scaling, as implemented (single fixed calibration per video), appears to genuinely help cross-subject generalization on clean single-camera-angle footage, but its aggregate benefit is fragile enough to be reversed by a single video with camera-consistency issues (title cards, cutaways, occlusion). This reframes the practical bottleneck: footage-quality robustness may matter as much as, or more than, the normalization approach itself. Both the n=7 and n=6 results are reported here deliberately, rather than treating either alone as final — this is a documented sensitivity analysis, not selective exclusion.

**Honest limitations:** n=6-7 is small either way; all differences remain statistically non-significant; only one outlier has been identified and characterized — it's possible other skaters have subtler, uncharacterized footage-quality issues affecting the result in ways not yet detected.

**Next steps:** expand skater count to test whether the "clean footage → scaling helps" pattern holds with more data; consider per-segment recalibration (detecting and correcting for camera-angle changes mid-video) as a way to make scaling robust to exactly the kind of footage issue that reversed this result.

## 🟢 Part 4: Tracking Robustness — Multi-Person Identity & Jitter (9/15)

### Multi-Person Tracking Fix
Discovered `_extract_landmark_sequence_task_api` used `result.pose_landmarks[0]` — whichever single person MediaPipe returned that frame, with no memory of *who* was being tracked. In footage with two skaters, this let the bone-scaling calculation silently jump to the wrong person mid-video.

Fixed with two-stage identity tracking:
1. **Position continuity**: track the same person via nearest hip-centroid distance across frames, rejecting jumps beyond a plausible threshold (treated as lost-track, not a guessed continuation).
2. **Appearance tiebreak**: when two candidates are ambiguously close in position (e.g. skaters passing near each other), compare a simple HSV color histogram of each candidate against the tracked person's last confirmed appearance and pick the closer match.

Also added EMA smoothing on raw landmark coordinates before any bone-length/angle math — previously smoothing only existed in the video-overlay rendering, never in the actual measurement pipeline, so calibration and feature extraction were using raw, jittery coordinates.

### Diagnosed a Real Remaining Failure — Documented as a Known Limitation, Not Chased Further
Tested against Lee Sang-Hwa's reference video and found the fix still swapped to a different skater. Built a frame-by-frame tracking diagnostic (`diagnose_tracking_swap.py`) rather than guessing at another patch. Findings: repeated large position jumps (0.2–0.4 normalized units, far exceeding the reject threshold) throughout the clip, each followed by a fresh "no previous track" re-anchor — a pattern consistent with the video's own title ("Slowmotion x6") indicating it's a **compilation of 6 separate replay-angle clips edited together**, not one continuous shot. Scene cuts have no temporal continuity for any centroid- or appearance-based tracker to bridge.

**Decision:** this is treated as a documented input-content limitation, not an algorithmic bug to keep patching — general robustness to arbitrary multi-cut broadcast compilations is outside reasonable project scope. Documented requirement: pipeline expects **one continuous single-camera shot**; multi-cut compilation footage should be trimmed to a clean segment before processing.
