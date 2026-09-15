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

## 💡 Real Next Steps
1. Expand the LOSO ablation beyond n=7 skaters.
2. Test per-segment recalibration (detecting camera-angle changes and recalibrating within a video) instead of one fixed scale per whole video.
3. Add a three-way comparison: bone-length scaling vs. no normalization vs. simple z-score standardization, as a proper baseline.
4. Review `multi_view_fusion.py` contents before re-claiming multi-camera fusion as working.
5. Get an actual measured latency comparison before re-claiming ONNX benefits, or drop the claim entirely if not pursued further.
