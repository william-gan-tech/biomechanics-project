# 🏆 Project Milestone & Achievement Log

> **Verification note (added 9/12):** This file was corrected after an audit
> found the original Phase 2 entry claimed completed ONNX edge inference
> that was never actually integrated into the running application. See
> `capabilities_phase2.md` and `abilities_phase3.md` for full details on
> what was corrected and why.

## Phase 1: Proof of Concept & Foundational Pipeline
*Research Question: To what extent can deep-learning architectures utilize comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation prior to observable athletic deceleration in elite speed skaters?*

* **Core Architecture:** Developed 3D spatial coordinate mapping using MediaPipe and joint-angle calculation frameworks.
* **Signal Processing:** Implemented digital Butterworth low-pass filtering to eliminate camera jitter and high-frequency noise.
* **Unsupervised Deep Learning:** Trained PyTorch autoencoders on fresh speed skating baseline data to detect form breakdown through reconstruction loss (MSE).
* **Predictive Validation:** Successfully tested lead-time tracking to demonstrate anomaly spikes occurring prior to observable athletic deceleration on segmented single-subject data.

## Phase 2: Automation & UI Deployment
* **Automated Video Ingestion:** Built `pipeline_engine.py` to process raw, unsegmented MP4 video files from start to finish.
* **Edge Acceleration — CORRECTED 9/12:** Original entry claimed completed ONNX quantization and edge runtime deployment. Audit found `onnxruntime` is imported in the dashboard but never actually invoked anywhere in the running code, and the "int8" quantized model file (626KB) is larger than the original FP32 file (133KB) with mixed float/int tensor types — indicating quantization was attempted but did not cleanly complete. **Status: attempted, not functional.**
* **Advanced Dashboard:** Deployed a feature-rich Streamlit web application (`dashboard.py`/`app.py`) with persistent session states (`st.session_state`), dynamic anomaly threshold sliders, and automated CSV reporting.
* **Execution Standardization:** Adopted Python's `-m` module execution flag to guarantee absolute path safety and eliminate relative import errors across environments.

## Phase 3: Cross-Subject Generalization (In Progress)
*Research Question: To what extent can relative bone-length scaling and proportional joint coordinate normalization improve cross-subject generalization in deep learning autoencoders to accurately detect neuromuscular fatigue across diverse athletes with distinct stylistic variances?*

* **Pipeline Reliability (9/08):** Fixed a deprecated MediaPipe API dependency (`mp.solutions.pose` no longer exists in the installed library version) and two silent path-resolution bugs that had been breaking dataset/config lookups since initial deployment.
* **Real Cross-Skater Comparison (9/09–9/10):** Replaced randomly-generated demo data in the Cross-Skater Anomaly dashboard mode with a real feature-extraction and DTW-comparison pipeline (`cross_skater_compare.py`), expanded to 7 of 10 tracked skaters with real video.
* **Leave-One-Skater-Out Ablation (9/11–9/12):** Built and ran a real cross-subject ablation comparing bone-length scaling against an unscaled baseline, using actual distinct athletes (not synthetic pseudo-subjects). Diagnosed and fixed two genuine data-quality failures surfaced during testing (single-leg occlusion miscalibration; a mid-clip broadcast camera cutaway).
* **Current Result (n=7 skaters, preliminary):** Bone-length scaling did NOT reduce cross-subject reconstruction-loss variance in this pilot (0.030 unscaled vs. 0.216 scaled) — recorded as the genuine result rather than adjusted to match the original hypothesis. See `abilities_phase3.md` for full methodology, limitations, and next steps.
* **Documentation Audit (9/12):** Reviewed all prior capability logs, journal entries, and the hours log against actual code; retracted a previously-claimed "100% variance reduction" result after finding the script that produced it tested time-chunks of a single video rather than distinct subjects, and
