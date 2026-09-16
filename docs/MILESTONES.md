# 🏆 Project Milestone & Achievement Log

> **Verification note (added 9/15):** This file was updated to reflect work
> through 9/15, correcting a Phase 3 entry that cited only the pre-outlier-
> sensitivity-check numbers as if they were the final result. See
> `CAPABILITIES_PHASE3.md` for full methodology, all four experiment
> scripts, and the complete statistical writeup.

## Phase 1: Proof of Concept & Foundational Pipeline
*Research Question: To what extent can deep-learning architectures utilize comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation prior to observable athletic deceleration in elite speed skaters?*

* **Core Architecture:** Developed 3D spatial coordinate mapping using MediaPipe and joint-angle calculation frameworks.
* **Signal Processing:** Implemented digital Butterworth low-pass filtering to eliminate camera jitter and high-frequency noise.
* **Unsupervised Deep Learning:** Trained PyTorch autoencoders on fresh speed skating baseline data to detect form breakdown through reconstruction loss (MSE).
* **Predictive Validation:** Successfully tested lead-time tracking to demonstrate anomaly spikes occurring prior to observable athletic deceleration on segmented single-subject data.

## Phase 2: Automation & UI Deployment
* **Automated Video Ingestion:** Built `pipeline_engine.py` to process raw, unsegmented MP4 video files from start to finish.
* **Edge Acceleration — CORRECTED 9/12:** Original entry claimed completed ONNX quantization and edge runtime deployment. Audit found `onnxruntime` is imported in the dashboard but never actually invoked anywhere in the running code, and the "int8" quantized model file (626KB) is larger than the original FP32 file (133KB) with mixed float/int tensor types — indicating quantization was attempted but did not cleanly complete. **Status: attempted, not functional.**
* **Advanced Dashboard:** Deployed a feature-rich Streamlit web application (`app.py`) with persistent session states (`st.session_state`), dynamic anomaly threshold sliders, and automated CSV reporting.
* **Execution Standardization:** Adopted Python's `-m` module execution flag to guarantee absolute path safety and eliminate relative import errors across environments.
* **Baseline Calibration — extended 9/15:** See Phase 3 entries below; calibration robustness work continued well past initial Phase 2 completion.

## Phase 3: Cross-Subject Generalization (Active)
*Research Question: To what extent can relative bone-length scaling and proportional joint coordinate normalization improve cross-subject generalization in deep learning autoencoders to accurately detect neuromuscular fatigue across diverse athletes with distinct stylistic variances?*

* **Pipeline Reliability (9/08):** Fixed a deprecated MediaPipe API dependency (`mp.solutions.pose` no longer exists in the installed library version) and two silent path-resolution bugs that had been breaking dataset/config lookups since initial deployment.
* **Real Cross-Skater Comparison (9/09–9/10):** Replaced randomly-generated demo data in the Cross-Skater Anomaly dashboard mode with a real feature-extraction and DTW-comparison pipeline (`cross_skater_compare.py`), expanded to 7 of 10 tracked skaters with real video.
* **Leave-One-Skater-Out Ablation — Phase 3a (9/11–9/12):** Built and ran a real cross-subject ablation comparing bone-length scaling against an unscaled baseline, using actual distinct athletes (not synthetic pseudo-subjects). Diagnosed and fixed two genuine data-quality failures surfaced during testing (single-leg occlusion miscalibration; a mid-clip broadcast camera cutaway).
* **Fatigue-Separability Ablation — Phase 3b (9/13):** Built a second, targeted experiment (`run_fatigue_separability_ablation.py`) that directly tests fatigue detection specifically — training only on other skaters' fresh-session data, then measuring the reconstruction-loss gap on a held-out skater's fatigued-session data. Phase 3a alone never distinguished fresh from fatigued motion.
* **Statistical Analysis Layer (9/13):** Built `analyze_phase3_results.py` — paired Wilcoxon signed-rank tests across both 3a and 3b. All comparisons at n=7 came back statistically non-significant (p > 0.4 throughout), reported honestly rather than treated as a confirmed result either way.
* **Outlier Sensitivity Check — Key Finding (9/14):** Noticed one skater's results dominated every comparison in both 3a and 3b. Built `outlier_sensitivity_check.py` to re-run all four key comparisons with that skater excluded. **Result: the finding reverses.** With all 7 skaters, scaling showed 3–10x higher cross-subject variance than unscaled across every comparison. Excluding the one skater with known camera-cutaway footage (n=6), scaling showed *lower* variance than unscaled in 3 of 4 comparisons. Both results are reported side by side — this is a documented sensitivity analysis, not selective exclusion.
* **Working interpretation:** Bone-length scaling, as implemented (single fixed calibration per video), may genuinely help cross-subject generalization on clean, single-camera-angle footage, but its aggregate benefit is fragile enough to be reversed by one video with real-world camera inconsistencies. Footage-quality robustness may matter as much as the normalization approach itself.
* **Tracking Robustness (9/15):** Fixed a real, user-reported bug — bone-scaling calibration could silently swap to a different skater mid-video in multi-person footage. Added position-continuity + appearance-histogram identity tracking, plus EMA smoothing on raw landmarks (previously only applied to video overlay rendering, not the actual measurement pipeline). Diagnosed a remaining failure case (a multi-cut compilation video) using a purpose-built frame-level diagnostic rather than guessing at further patches; documented as a known input-content limitation rather than chased indefinitely.
* **Documentation Audit (9/12, extended 9/14–9/15):** Reviewed all prior capability logs, journal entries, and the hours log against actual code. Retracted a previously-claimed "100% variance reduction" result after finding the script that produced it tested time-chunks of a single video rather than distinct subjects, and contained a self-recursive `main()` call that would prevent it from ever completing a clean run. Downgraded ONNX and multi-camera fusion claims from "completed" to "attempted/unverified." Cleaned up repository structure (untracked committed temp/cache video files, archived ~60 legacy exploratory scripts, removed a nested duplicate folder).

## 🎯 Where the Project Actually Stands (as of 9/15)
* **Phase 1:** Genuinely completed and validated.
* **Phase 2:** Mostly completed — ingestion, dashboard, and calibration work; ONNX edge acceleration attempted but not functional.
* **Phase 3:** Active, with two real experiments (3a, 3b), proper statistical testing, and a genuinely interesting sensitivity finding — the result's direction depends on footage quality, which is itself a defensible, reportable conclusion. Tracking robustness (multi-person identity, jitter smoothing) extended 9/15 with one documented remaining limitation.
* **Phase 4 (proposed):** Start-phase biomechanics — testing whether bone-scaled trajectories distinguish explosive start acceleration from steady-state cruising, and whether the 9/15 multi-person tracking system generalizes to skaters genuinely sharing a start line (not just sequential video cuts).
