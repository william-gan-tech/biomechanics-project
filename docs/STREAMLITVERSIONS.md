# 🧊 Biomechanics Project Version History & Progress

> **Verification note (added 9/12):** Version 3.0's ONNX claim was
> downgraded after audit (see `capabilities_phase2.md`). A new Version 3.1
> entry has been added reflecting real work completed 9/08–9/12, including
> a documentation correction pass after several claims in earlier versions
> could not be verified against the actual codebase.

## Version 1.0: Core Pipeline & Initial Framework
* **Baseline Architecture:** Established the core video processing pipeline and initial skeleton keypoint extraction using MediaPipe 3D pose estimation.
* **Basic Visualization:** Implemented raw reconstruction loss tracking and matplotlib charts for foundational biomechanical analysis.
* **Initial Setup:** Configured local testing structure, dependency tracking (`requirements.txt`), and initial baseline analysis for speed skating form.

## Version 2.0: Advanced Analytics & UI Expansion
* **Multi-Mode Dashboard:** Upgraded the Streamlit interface to support specialized analytical modes.
* **Automated Stride Segmentation:** Added kinematic stride splitting and knee-angle profile overlays.
* **Predictive Lead-Time Analysis:** Introduced early fatigue anomaly detection algorithms with CSV reporting.
* **Comparative Insights:** Enabled fresh-vs-fatigued segment comparison on real single-subject data.
* **Known limitation (flagged 9/12):** The Cross-Skater Anomaly and Form/Technique baseline modes introduced in this version displayed hardcoded placeholder values and seeded-random demo data rather than computed measurements. This was not caught at the time and persisted through Version 3.0.

## Version 3.0: Edge Optimization Attempt & Cloud Deployment
* **ONNX Runtime Integration — CORRECTED 9/12:** Original entry claimed working ONNX edge inference. Audit found `onnxruntime` is imported but never called in the running application; the quantized model file is larger than the original with mixed tensor types, indicating quantization did not complete cleanly. **This feature is not currently functional.**
* **Strict Content Validation:** Implemented automated checks parsing video metadata and confidence scores to gate input processing.
* **Streamlit Cloud Pipeline Integration:** Connected the GitHub repository to Streamlit Community Cloud for continuous deployment.
* **Interactive UI & Auto-Digest Mode:** Polished dashboard visuals; introduced automated video ingestion and upload workflow (`Auto-Digest`) — this mode genuinely runs real MediaPipe extraction and autoencoder inference, unlike the Cross-Skater/Form-Technique modes noted above.
* **Professional Feedback System:** Added multi-metric analytics cards, interactive threshold adjustments, and downloadable CSV summary reports.

## Version 3.1: Real Cross-Subject Validation & Documentation Audit (Current)
* **Fixed Pipeline Bugs (9/08):** Migrated off a removed MediaPipe API; fixed two silent path-resolution bugs breaking dataset lookups since launch.
* **Replaced Placeholder Data with Real Computation (9/09):** The Cross-Skater Anomaly mode flagged as a known limitation in v2.0 was rebuilt from scratch — real feature extraction, disk caching, and DTW-based comparison across actual skater video, replacing all `np.random`-generated demo data in that mode.
* **Expanded Real Video Coverage (9/10):** Downloaded and integrated footage for 4 additional skaters, bringing real-video cross-skater comparison coverage to 7 of 10 tracked athletes.
* **Real Leave-One-Skater-Out Ablation (9/11–9/12):** Built a genuine cross-subject experiment (not synthetic pseudo-subjects) directly testing whether bone-length scaling improves generalization. Diagnosed and fixed two real data-quality bugs found during testing. Current preliminary result: bone-scaling increased cross-subject loss variance rather than reducing it (n=7, see `abilities_phase3.md`) — reported as the genuine finding, not adjusted to fit expectations.
* **Full Documentation Audit (9/12):** Systematically checked prior milestone claims against actual code and git history. Retracted a previously-claimed "100% variance reduction" result (the script producing it tested arbitrary time-chunks of one video, not distinct subjects, and contained a bug preventing successful completion). Downgraded ONNX and multi-camera fusion claims from "completed" to "attempted/unverified" across all tracking documents.
* **Outstanding items:** `requirements.txt` needs regeneration from a real `pip freeze` (previous version contained implausible package versions, e.g. a nonexistent OpenCV 5.x release). `multi_view_fusion.py` contents not yet reviewed to confirm real functionality.
