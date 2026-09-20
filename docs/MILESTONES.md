# 🏆 Project Milestone & Achievement Log

> **Verification note (added 9/12, updated 9/18):** This file has been
> kept in sync with verified progress through Phase 4. See
> `docs/CAPABILITIES_PHASE1.md` through `docs/CAPABILITIES_PHASE4.md`
> and `docs/PHASE1_SUMMARY.md` through `docs/PHASE4_SUMMARY.md` for full
> per-phase detail and methodology.

## Phase 1: Proof of Concept & Foundational Pipeline
*Research Question: To what extent can deep-learning architectures utilize comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation prior to observable athletic deceleration in elite speed skaters?*

* **Core Architecture:** Developed 3D spatial coordinate mapping using MediaPipe and joint-angle calculation frameworks.
* **Signal Processing:** Implemented digital Butterworth low-pass filtering to eliminate camera jitter and high-frequency noise.
* **Unsupervised Deep Learning:** Trained PyTorch autoencoders on fresh speed skating baseline data to detect form breakdown through reconstruction loss (MSE).
* **Predictive Validation:** Successfully tested lead-time tracking to demonstrate anomaly spikes occurring prior to observable athletic deceleration on segmented single-subject data.

## Phase 2: Automation & UI Deployment
* **Automated Video Ingestion:** Built `pipeline_engine.py` to process raw, unsegmented MP4 video files from start to finish.
* **Edge Acceleration — corrected 9/12, resolved (partially) 9/16:** ONNX FP32 export is now genuinely functional and verified: numerically equivalent to PyTorch (max difference 0.000031) and **measured 3.08x faster** via a real reproducible benchmark. INT8 quantization's file-size bug was fixed (144KB vs. 495KB, a real 70.9% reduction) but a genuine correctness bug remains (max output difference 55.7 vs. PyTorch) — documented honestly as not currently usable rather than claimed as working.
* **Advanced Dashboard:** Deployed a feature-rich Streamlit web application (`app.py`) with persistent session states (`st.session_state`), dynamic anomaly threshold sliders, and automated CSV reporting.
* **Execution Standardization:** Adopted Python's `-m` module execution flag to guarantee absolute path safety and eliminate relative import errors across environments.

## Phase 3: Cross-Subject Generalization (Complete)
*Research Question: To what extent can relative bone-length scaling and proportional joint coordinate normalization improve cross-subject generalization in deep learning autoencoders to accurately detect neuromuscular fatigue across diverse athletes with distinct stylistic variances?*

* **Pipeline Reliability (9/08):** Fixed a deprecated MediaPipe API dependency and two silent path-resolution bugs that had been breaking dataset/config lookups since initial deployment.
* **Real Cross-Skater Comparison (9/09–9/10):** Replaced randomly-generated demo data in the Cross-Skater Anomaly dashboard mode with a real feature-extraction and DTW-comparison pipeline (`cross_skater_compare.py`), expanded to 7 of 10 tracked skaters with real video.
* **Leave-One-Skater-Out Ablation — Phase 3a (9/11–9/12):** Real cross-subject ablation using actual distinct athletes. Diagnosed and fixed two genuine data-quality failures (single-leg occlusion miscalibration; a mid-clip broadcast camera cutaway).
* **Fatigue-Separability Ablation — Phase 3b (9/13):** A second, more targeted experiment testing fatigue detection specifically — training on other skaters' fresh data, measuring reconstruction-loss gap on a held-out skater's fatigued data.
* **Statistical Analysis Layer (9/13):** Paired Wilcoxon signed-rank tests across both experiments — all comparisons at n=7 non-significant (p > 0.4), reported honestly.
* **Outlier Sensitivity Check — Key Finding (9/14):** Found the headline "scaling increases variance" result was driven almost entirely by one camera-cutaway-affected skater. **Result reverses when excluded** (n=6): scaling shows *lower* variance than unscaled in 3 of 4 comparisons. Both versions reported side by side.
* **Tracking Robustness (9/15):** Fixed a real bug — bone-scaling calibration could silently swap to a different skater mid-video in multi-person footage. Added position-continuity + appearance-histogram identity tracking, plus EMA smoothing on raw landmarks.
* **Documentation Audit (9/12, extended through 9/15):** Retracted a previously-claimed "100% variance reduction" result after finding the producing script tested arbitrary time-chunks of one video, not distinct subjects, and contained a bug preventing a clean run. Downgraded ONNX and multi-camera fusion claims from "completed" to "attempted/verified as partial." Cleaned up repository structure.

## Phase 4: Start-Phase, Corner, and Straightaway Analysis (Complete, 9/16–9/18)
*Research Question: To what extent can bone-length-scaled joint-angle trajectories distinguish explosive start-phase acceleration mechanics from steady-state cruising form, and does the multi-person identity-tracking approach generalize to reliably isolating a single target skater from simultaneous competitors at a shared start line?*

* **Footage Audit (9/16):** Checked existing footage for genuine multi-person content before sourcing new video — found 5 of 7 skaters had some, though later found to be mostly incidental, not true start-line content.
* **Feature Engineering (9/16):** Added torso-lean/crouch angle, hip velocity, and hip acceleration — new biomechanical features not present in the Phase 3 feature set.
* **Automated Detection Failure, Documented Honestly (9/18):** Acceleration-spike detection found zero valid start candidates across two real Olympic videos, due to landmark jitter on chaotic broadcast footage. Caught and corrected two false positives where a "biggest spike" turned out to be a broadcast graphic overlay, not real motion.
* **Real Start Confirmed via Manual Verification (9/18):** Found and visually confirmed a genuine start sequence in real Olympic short-track footage (`start_candidate_3.mp4`) after automated detection failed — held REST position, a documented camera-cut data gap, and confirmed EARLY-ACCELERATION phase.
* **Identity Question Resolved (9/18):** Two independent checks (appearance histogram, then spatial gap-continuity) confirmed the REST and EARLY-ACCELERATION segments show the same skater — the torso-lean sign flip between them is a real signal, not a tracking artifact.
* **Real Within-Subject Technique-Phase Finding (9/18):** Compared start, corner, and straightaway phases in the same skater. **Torso-lean stability: straightaway (26.2° std) > start (35.9°) >> corner (82.9°)** — corners 3.2x more variable, matching known cornering biomechanics. **Knee asymmetry: straightaway nearly symmetric (2° gap), corner clearly asymmetric (14° gap)** — real evidence motivating the need for bilateral tracking in future corner-focused work.
* **Multi-Person Tracking Validated Under Real Pack Conditions (9/18):** Ran the full tracking diagnostic against genuine Olympic pack-racing footage — 135 real ambiguous multi-person events, 10-sample manual review confirmed correct identity resolution every time.

## 🎯 Where the Project Actually Stands (as of 9/18)
* **Phase 1:** Genuinely completed and validated.
* **Phase 2:** Mostly completed — ONNX FP32 acceleration now genuinely real and measured (3.08x); INT8 remains a documented, unresolved limitation.
* **Phase 3:** Complete — two real experiments, proper statistics, and a genuinely interesting sensitivity finding (result direction depends on footage quality).
* **Phase 4:** Complete — both sub-questions answered with real, physically sensible evidence from genuine, visually-verified Olympic footage.
* **Phase 5 (planned):** Straightaway stroke mechanics — already has a real preliminary baseline from Phase 4's technique-phase comparison.
* **Phase 6 (planned):** Corner technique — already has a real, evidence-backed reason (measured knee asymmetry) that bilateral tracking will be necessary.