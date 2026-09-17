# 🧊 Deep Learning Biomechanics & Injury Prevention Engine
*Advanced Temporal Trajectory Analysis for Speed Skating Form Breakdown & Fatigue Prediction*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2F1GPU-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](docs/LICENSE)

> **Documentation integrity note:** On 9/12, this project underwent a full
> documentation audit after several previously-claimed results (a "100%
> variance reduction," working ONNX edge inference, multi-camera fusion
> status) could not be corroborated against the actual codebase or were
> directly contradicted by it. Those claims were retracted or downgraded —
> see `docs/CAPABILITIES_PHASE3.md` and `docs/JOURNAL.md` for the full
> record of what was found and corrected. Everything below reflects only
> what has been independently verified by actually running the code.

---

## 🎯 Formal Research Question

> **To what extent can deep learning models leverage comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation prior to observable athletic deceleration in elite speed skaters?**

This umbrella question is broken into phase-specific sub-questions as the project has progressed — see **Core Project Status** below.

---

## A Little Self Introduction

> As someone deeply immersed in sports, AI, and robotics, I decided to build this model because I am a speed skater myself, having competed internationally and earned national gold medals representing Team USA, while also sharing a background in endurance sports like cross country, track, and running half marathons in San Jose and San Francisco. Between experiencing those sports firsthand and achieving a #11 world ranking in VEX robotics, I've always been fascinated by how technology and athletics intersect. I watched runners and skaters struggle with fatigue, noticing that humans usually only spot form breakdown — like skaters not bending their knees, or runners losing proper posture — after it has already happened and they are already slowing down. That sparked my core question: to what extent can deep learning models leverage temporal joint-angle trajectories to anticipate biomechanical performance degradation prior to measurable athletic deceleration in speed skaters? By combining my athletic background with my love for AI and robotics, I wanted to build something that moves past reactive observation into true predictive intelligence.

---

## 💡 Why This Project Matters

Traditional sports biomechanics relies on subjective human observation or expensive, fixed laboratory motion-capture equipment. This project builds an automated, accessible alternative:

* **Proactive Injury Prevention:** The system learns an individual skater's normal movement baseline during fresh runs and flags mechanical drift as it starts to occur.
* **Objective Coaching Intelligence:** Compares developing athletes against elite reference forms (Sven Kramer, Jorrit Bergsma, Haralds Silovs, Patrick Meek, Ragne Wiklund, Mia Manganello Kilburg, Jan Blokhuijsen) with real, computed kinematic data rather than guesswork.
* **Early-Warning Capability:** Investigates whether temporal trajectory analysis can catch mechanical breakdown before it's visible as physical deceleration.

---

## ⚡ Core Project Status & Architecture

### Phase 1 — Completed
Proved that deep learning autoencoders and LSTM architectures can utilize comparative temporal joint-angle trajectories across manually-segmented clips to distinguish fresh from fatigued movement on a single subject, before observable athletic deceleration.

### Phase 2 — Mostly Completed
Automated end-to-end video ingestion (`pipeline_engine.py`), a working Streamlit dashboard (`app.py`) with persistent session state, and automated baseline calibration are functional and verified.

**ONNX edge acceleration — partially functional, corrected 9/16:**
- **FP32 export: genuinely functional.** Rebuilt the export pipeline (the previous version used PyTorch's newer "dynamo" exporter, which produced shape mismatches incompatible with this LSTM architecture; switched to the legacy exporter). Verified numerically equivalent to the PyTorch model (max output difference: 0.000031) and **measured 3.08x faster inference** than PyTorch (0.918ms vs. 2.824ms mean, batch size 8, 200 runs, CPU) via a reproducible benchmark (`benchmark_onnx_speed.py`). `onnx_inference.py` provides a real `ONNXFatigueDetector` class that actually calls `onnxruntime.InferenceSession` — previously `onnxruntime` was imported but never invoked anywhere in the codebase.
- **INT8 quantization: file-size bug fixed, but not usable.** The previous "int8" file was larger than the FP32 original (626KB vs. 133KB); now genuinely smaller (144KB vs. 495KB, a real 70.9% reduction). However, output verification shows a max difference of 55.7 vs. PyTorch — dynamic INT8 quantization of this model's LSTM recurrent weight matrices produces functionally broken output, not a normal accuracy/speed tradeoff. **Not currently usable for inference**, documented as a known limitation of dynamic quantization on LSTM architectures rather than claimed as working.

See `docs/CAPABILITIES_PHASE2.md` for full detail.

### Phase 3 — Active, with real, statistically-analyzed findings

> *Phase 3: To what extent can relative bone-length scaling and proportional joint coordinate normalization improve cross-subject generalization in deep learning autoencoders to accurately detect neuromuscular fatigue across diverse athletes with distinct stylistic variances?*

Split into two sub-experiments for precision:

- **Phase 3a** — *Does bone-length scaling reduce cross-subject variance in general motion-reconstruction loss?* Leave-One-Skater-Out ablation across 7 real distinct athletes (`run_bone_scaling_ablation.py`), reproduced twice with consistent results.
- **Phase 3b** — *Does bone-length scaling improve an autoencoder's ability to separate fresh from fatigued movement in an athlete unseen during training?* (`run_fatigue_separability_ablation.py`) — this is the experiment that actually matches the fatigue-detection wording of the original question; 3a alone does not test fatigue detection.

**Key finding:** With all 7 skaters, bone-length scaling showed 3–10x *higher* cross-subject variance than an unscaled baseline across every comparison in both 3a and 3b. An outlier sensitivity check (`outlier_sensitivity_check.py`) found this result was driven almost entirely by one athlete whose footage contains a mid-clip broadcast camera cutaway (previously diagnosed and partially, but not fully, mitigated). **Excluding that one athlete (n=6), the result reverses**: scaling shows *lower* variance than the unscaled baseline in 3 of 4 key comparisons.

Paired Wilcoxon signed-rank tests (`analyze_phase3_results.py`) on all comparisons were **not statistically significant** at this sample size (p > 0.4 throughout) — reported honestly rather than treated as either a confirmed positive or null result.

**Working interpretation:** Bone-length scaling, as implemented (a single fixed calibration per video), may genuinely help cross-subject generalization on clean, single-camera-angle footage, but its aggregate benefit is fragile enough to be reversed by one video with real-world camera inconsistencies. Robustness to footage quality may matter as much as, or more than, the normalization approach itself. Full methodology, all four experiment scripts, and the complete statistical writeup are in `docs/CAPABILITIES_PHASE3.md`.

Multi-camera stream fusion (`multi_view_fusion.py`) contains real, sensible interpolation/merge logic but is **not currently integrated** into the running pipeline, has one identified bug (row-position interpolation instead of true time-based interpolation), and has only been tested against a 4-row synthetic mock — not real footage.

---

## 📊 Verified Pipeline & Dashboard Capabilities

* **🎥 Automated Video Ingestion (`pipeline_engine.py`):** Processes raw MP4 files and YouTube URLs end-to-end via `yt_dlp` — extracts pose landmarks, computes joint angles, runs autoencoder reconstruction loss.
* **🤖 AI-Powered Pose Estimation:** MediaPipe Tasks `PoseLandmarker` API (migrated 9/08 off the removed legacy `mp.solutions.pose` API).
* **🦴 Bone-Length Calibration (`compute_video_reference_scale`):** Samples early frames per video, automatically skips intro/title-card footage, and picks whichever leg is visible per frame (fixed 9/12 after diagnosing an occlusion-driven calibration failure).
* **📐 Biomechanical Angle Calculation:** 3D Euclidean coordinates from MediaPipe used to compute real joint angles per frame.
* **📉 Signal Noise Reduction:** Digital Butterworth low-pass filtering on raw joint-angle time series.
* **👥 Real Cross-Skater Comparison (`cross_skater_compare.py`):** DTW-based comparison of real bone-scaled joint trajectories across skaters, with an honest tiered fallback (video → CSV knee-angle-only → clearly-labeled simulated) instead of silently faking results.
* **🏁 Real Fresh-vs-Fatigued Analysis:** Dashboard's 3000m comparison mode computes real stride frequency and knee-angle variability from actual session data where available (Patrick Meek, Mia Manganello Kilburg), with an honest simulated fallback for skaters without usable data.
* **🖥️ Interactive Web Dashboard (`app.py`):** Persistent session state, live threshold sliders, auto-digest mode, dynamic skater selection, CSV report export.
* **🎬 Annotated Video Rendering:** Outputs downloadable `.mp4` with skeleton overlays and live bone-scaling readouts.
* **🧠 Unsupervised Deep Learning Anomaly Detection:** PyTorch LSTM autoencoder trained on fresh baseline movement, scored via reconstruction MSE.
* **📊 Statistical Ablation Framework:** Leave-One-Skater-Out cross-validation with training-pool-only standardization, incremental/resumable result saving, and paired significance testing (Wilcoxon signed-rank).

---

## 🚀 Getting Started & Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
python -m streamlit run app.py

# Run the bone-scaling ablation (Phase 3a)
python -m run_bone_scaling_ablation

# Run the fatigue-separability ablation (Phase 3b)
python -m run_fatigue_separability_ablation

# Run the combined statistical analysis
python -m analyze_phase3_results
```

Execute scripts using Python's `-m` module flag to avoid relative import path issues.

---

## 🔮 Phase 4 Roadmap (Planned, Not Yet Started)

* **Footage-Robustness Testing:** Directly follow up on the Phase 3 outlier finding — test whether per-segment recalibration (detecting and correcting for mid-video camera-angle changes) makes bone-length scaling robust to the kind of footage inconsistency that reversed the Phase 3a/3b result.
* **Expanded Sample Size:** Grow past n=7 skaters to get a more statistically powered answer to the Phase 3 question.
* **Real ONNX Edge Deployment:** Actually complete and integrate INT8 quantization (the current attempt did not cleanly reduce model size or get wired into live inference).
* **Multi-Camera Fusion Integration:** Fix the identified interpolation bug in `multi_view_fusion.py`, integrate it into `pipeline_engine.py`, and validate against real (not synthetic) multi-angle footage.
* **Temporal Smoothing & Confidence Gating:** EMA coordinate filtering and confidence-score masking to reduce landmark jitter during high-speed motion blur (scoped 9/06–9/07, not yet completed).

---

## 📈 Project Documentation

All tracking documents live in `docs/`:

* **`JOURNAL.md`** — daily engineering log, including a 9/12 correction pass on earlier entries
* **`HOURS.md`** — time tracking
* **`CAPABILITIES_PHASE1.md`**, **`CAPABILITIES_PHASE2.md`**, **`CAPABILITIES_PHASE3.md`** — verified capability logs per phase
* **`MILESTONES.md`** — high-level milestone summary
* **`phase3_statistical_summary.md`** (repo root) — auto-generated statistical report from `analyze_phase3_results.py`

---

## 🛠️ System Architecture

```text
[Raw Video / YouTube URL] → [pipeline_engine.py: yt_dlp + MediaPipe PoseLandmarker]
                                          │
                    [Bone-Length Calibration + Butterworth Filtering]
                                          │
                    [PyTorch LSTM Autoencoder: Reconstruction Loss]
                                          │
        [Streamlit Dashboard: Metrics, Charts, CSV Export, Cross-Skater Comparison]
```

---

## 🧹 Repository Notes

This repo currently has some cleanup pending: several `temp_*.mp4`/`downloaded_skater.*`/`rendered_skating_output*.mp4` files were committed that should be gitignored (ephemeral pipeline output, not source), and a nested duplicate folder should be verified and removed. Tracked as a housekeeping item alongside Phase 4 planning.
