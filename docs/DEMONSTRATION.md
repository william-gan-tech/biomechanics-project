# ⚡ Biomechanics & Deep Learning Anomaly Detection: System Demonstration Guide (`demonstration.md`)

> **Correction & Update (added 9/12, extended 9/18):** The original
> screenshots and metrics in this document for Dashboard Modes 1–4
> (Cross-Skater Generalization, 3000m Fresh vs. Fatigued, Form &
> Technique Baseline, First-Ever Baseline Analysis) were captured from an
> earlier version of the dashboard that displayed hardcoded placeholder
> values and seeded-random demo data, NOT real computed biomechanical
> measurements.
>
> **Current status per mode:**
> - **Mode 1 (Cross-Skater Comparison):** ✅ Real, rebuilt 9/09 —
>   genuine DTW-based comparison using actual extracted joint trajectories.
> - **Mode 2 (3000m Fresh vs. Fatigued):** ✅ Real for skaters with usable
>   source data (Patrick Meek, Mia Manganello Kilburg), with an honest
>   labeled fallback for skaters without it (Carlijn, Sandrina).
> - **Modes 3–4:** Still use placeholder/simulated data as of this
>   writing — not yet corrected.
> - **Mode 5 (Auto-Digest):** Always real — runs the actual pipeline
>   end-to-end.
>
> This document needs new screenshots taken from the corrected dashboard
> before further public presentation. Old screenshots retained below only
> where still accurate (Mode 5, setup steps).

---

## 📋 Overview
This document outlines the operational execution, pipeline architecture,
and cloud deployment of the biomechanics anomaly detection system.
Extracts 3D joint kinematics from video, evaluates temporal movement
patterns via deep learning autoencoders, and surfaces insights through an
interactive dashboard — now extended (Phase 4) with start-phase,
corner-phase, and straightaway-phase biomechanical analysis and
multi-person identity tracking for footage with simultaneous skaters.

---

## Part 1: Initial Environment Setup & Prerequisites

### Step 1: Install Python 3.12+
Download from python.org. Check **"Add python.exe to PATH"** during
install.

### Step 2: Install Microsoft C++ Runtime
Required for PyTorch to load correctly:
[aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Step 3: Clone and Set Up
```bash
git clone https://github.com/william-gan-tech/biomechanics-project.git
cd biomechanics-project
pip install -r requirements.txt
```
`requirements.txt` was regenerated 9/12 from a real, verified `pip
freeze` — safe to trust as-is.

---

## 🌐 Launching the Interactive Streamlit Dashboard

> **Important — corrected 9/12:** The entry point is `app.py` at the
> project root, **not** `src/dashboard.py`. That file was archived during
> repository cleanup; running the old path will fail. If a deployed
> Streamlit Cloud instance still references `src/dashboard.py`, its
> "Main file path" setting needs updating to `app.py`.

```bash
python -m streamlit run app.py
```

---

## 📊 Verified Analytical & Interactive Features

* **Real Bone-Length Calibration:** Automatically skips intro/title-card
  frames, picks whichever leg is visible per frame (fixed 9/12 after
  diagnosing a real occlusion-driven calibration failure), and — as of
  9/15 — maintains correct skater identity across frames using position
  continuity and appearance-histogram matching when multiple people are
  in frame.
* **Real Cross-Skater Comparison (Mode 1):** DTW-based comparison of
  actual extracted joint trajectories, with honest tiered fallback (real
  video → CSV knee-angle-only → clearly labeled simulated) instead of
  silently faking results when data is unavailable.
* **Real Fresh vs. Fatigued Analysis (Mode 2, where data exists):** Real
  stride-frequency and knee-angle-variability computation from actual
  session data.
* **Dynamic Statistical Thresholding:** Automated sports-science
  boundary calculations, properly scaled per metric (fixed 9/09 after
  finding the original threshold slider was miscalibrated for the real
  DTW comparison metric).
* **ONNX Edge Acceleration — partially real, corrected 9/16:** FP32
  export is genuinely functional, numerically verified against PyTorch
  (max difference 0.000031), and **measured 3.08x faster** via a
  reproducible benchmark. INT8 quantization's file-size bug is fixed but
  a real correctness bug remains — not currently usable for inference,
  documented honestly rather than claimed as working.
* **Multi-Person Identity Tracking:** Validated 9/18 against genuine
  Olympic pack-racing footage — 135 real ambiguous multi-person events
  correctly resolved, sample-reviewed and confirmed accurate.
* **Start/Corner/Straightaway Technique Analysis (Phase 4, new):**
  Torso-lean angle, hip velocity, and hip acceleration features,
  distinguishing biomechanically distinct technique phases within a
  single skater's race — real finding: corners show 3.2x more torso-lean
  variability than straightaways, with measurable knee asymmetry.

---

## 🛠️ System Architecture & Workflow

### 1. Data Ingestion & Pose Estimation (`pipeline_engine.py`)
* Processes local MP4 files or YouTube URLs via `yt_dlp`.
* MediaPipe Tasks `PoseLandmarker` API (migrated 9/08 off the removed
  legacy `mp.solutions.pose` API).
* Computes 3D joint angles from real Euclidean coordinates.
* Detects and tracks up to 2 simultaneous people per frame, maintaining
  one consistent target identity across the whole clip.

### 2. Kinematic Processing
* Digital Butterworth low-pass filtering.
* Sliding-window feature extraction with real bone-length normalization
  (single calibrated scale per video — see `docs/CAPABILITIES_PHASE3.md`
  for the full ablation testing this approach's cross-subject
  generalization).
* Phase 4 additions: torso-lean angle, hip velocity/acceleration for
  start/corner/straightaway differentiation.

### 3. Deep Learning Anomaly Detection (PyTorch)
* Unsupervised LSTM autoencoder trained on fresh baseline movement.
* Reconstruction error (MSE) as the fatigue/anomaly signal.
* Real Leave-One-Skater-Out ablation testing (Phase 3a/3b) with proper
  statistical significance testing — see `docs/PHASE3_SUMMARY.md` for
  the full, honestly-reported result (including a real outlier
  sensitivity finding that reversed the headline conclusion).

---

## 🚀 Running the Demonstration Locally

```bash
cd biomechanics-project
pip install -r requirements.txt
python -m streamlit run app.py
```
Streamlit opens automatically at `http://localhost:8501`.

### Running Phase 3/4 Research Scripts
```bash
python -m run_bone_scaling_ablation           # Phase 3a
python -m run_fatigue_separability_ablation   # Phase 3b
python -m analyze_phase3_results              # Combined statistics
python -m compare_technique_phases            # Phase 4a start/corner/straightaway
python -m diagnose_tracking_swap              # Phase 4b tracking validation
```
Always run with Python's `-m` flag to avoid relative import errors.

---

## 🏆 Project Milestone Summary
Phase 1 and Phase 2 validated single-subject fatigue detection. Phase 3
tested cross-subject generalization of bone-length scaling with real,
statistically-analyzed results (a genuine, honestly-reported finding
that reversed upon outlier removal). Phase 4 extended the pipeline to
start-phase, corner-phase, and straightaway-phase analysis with real,
visually-verified footage, and validated multi-person identity tracking
under genuine pack-racing conditions. See `docs/PHASE1_SUMMARY.md`
through `docs/PHASE4_SUMMARY.md` for full per-phase writeups.