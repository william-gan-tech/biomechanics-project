# 🧊 Biomechanics Project Version History & Progress

> **Verification note (added 9/18):** This file has been kept current
> through Phase 4. Version 3.0's ONNX claim was downgraded 9/12, then
> partially resolved for real 9/16 (see Version 3.2). Phase 4 work is
> tracked as its own version (4.0) rather than folded into Version 3.x.
> See `docs/CAPABILITIES_PHASE2.md` through `docs/CAPABILITIES_PHASE4.md`
> for full per-phase detail.

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

## Version 3.1: Real Cross-Subject Validation & Documentation Audit
* **Fixed Pipeline Bugs (9/08):** Migrated off a removed MediaPipe API; fixed two silent path-resolution bugs breaking dataset lookups since launch.
* **Replaced Placeholder Data with Real Computation (9/09):** The Cross-Skater Anomaly mode flagged as a known limitation in v2.0 was rebuilt from scratch — real feature extraction, disk caching, and DTW-based comparison across actual skater video.
* **Expanded Real Video Coverage (9/10):** Downloaded and integrated footage for 4 additional skaters, bringing real-video cross-skater comparison coverage to 7 of 10 tracked athletes.
* **Real Leave-One-Skater-Out Ablation (9/11–9/12):** Genuine cross-subject experiment (not synthetic pseudo-subjects) directly testing whether bone-length scaling improves generalization.
* **Full Documentation Audit (9/12):** Retracted a previously-claimed "100% variance reduction" result (the script producing it tested arbitrary time-chunks of one video, not distinct subjects, and contained a bug preventing successful completion). Downgraded ONNX and multi-camera fusion claims from "completed" to "attempted/unverified."

## Version 3.2: Statistical Rigor & Real ONNX Fix
* **Fatigue-Separability Ablation & Statistical Layer (9/13):** Built a second Phase 3 experiment testing fatigue detection specifically (not just general variance), plus a paired Wilcoxon significance-testing layer across all results.
* **Outlier Sensitivity Check — Real Finding (9/14):** Found the Phase 3a/3b "scaling increases variance" result was driven almost entirely by one camera-cutaway-affected skater. **The result reverses when that skater is excluded** — both versions reported side by side rather than picking one.
* **Multi-Person Tracking Fix (9/15):** Fixed a real bug allowing bone-scaling calibration to silently swap identity between skaters in multi-person footage. Added position-continuity + appearance-histogram tracking and EMA jitter smoothing.
* **ONNX Genuinely Fixed — Partial Success (9/16):** Rebuilt export/quantization from scratch. **FP32 export is now real and verified**: numerically equivalent to PyTorch (max diff 0.000031), measured **3.08x faster** via a reproducible benchmark. INT8 quantization's file-size bug was fixed (144KB vs 495KB, genuine 70.9% reduction) but a real correctness bug was found and honestly documented as unresolved (max output difference 55.7 vs PyTorch) rather than claimed as working.

## Version 4.0: Phase 4 — Start, Corner, and Straightaway Analysis (Current, 9/16–9/18)
* **Extended Scope:** First version to go beyond fatigue/cross-subject analysis into technique-phase biomechanics — start-phase, corner, and straightaway mechanics within a single skater's race.
* **Automated Detection Failure, Documented Honestly (9/18):** Acceleration-spike start detection found zero valid candidates on real Olympic footage due to landmark jitter on chaotic broadcast video. Caught and corrected two separate false positives where a "biggest spike" turned out to be a broadcast graphic overlay, not real motion.
* **Real Start Confirmed via Manual Verification (9/18):** Pivoted to manual visual frame-by-frame review after automated detection failed; found and confirmed a genuine start sequence in real Olympic short-track footage.
* **Identity Question Resolved (9/18):** Two independent checks (appearance histogram, then spatial gap-continuity) confirmed a real, not artifactual, signal in the start-to-acceleration transition.
* **Real Within-Subject Technique-Phase Finding (9/18):** Torso-lean stability ranked straightaway (26.2° std) > start (35.9°) >> corner (82.9°) — corners 3.2x more variable, matching known cornering biomechanics. Knee asymmetry: 2° gap on straightaways vs. 14° gap in corners — direct, evidence-backed motivation for planned Phase 6's bilateral tracking requirement.
* **Multi-Person Tracking Validated Under Real Pack Conditions (9/18):** 135 real ambiguous multi-person events across genuine Olympic pack-racing footage, 10-sample manual review confirmed correct identity resolution every time — the first real validation of the 9/15 tracking fix under its actual intended use case.
* **Repository & Documentation Consolidation (9/12–9/18):** Cleaned up repository structure, added `FIXES.md` as a consolidated correction log, and added per-phase narrative summaries (`PHASE1_SUMMARY.md` through `PHASE4_SUMMARY.md`).
* **Outstanding items:** ONNX INT8 quantization remains genuinely broken (not just unintegrated) — would need static/calibration-based quantization to fix. `multi_view_fusion.py` remains real but unintegrated. Phase 4's findings are n=1 (one confirmed real start video) — real and rigorous within that scope, not yet cross-athlete generalizable.