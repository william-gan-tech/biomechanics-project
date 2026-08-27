# 🧊 Deep Learning Biomechanics & Injury Prevention Engine
*Advanced Temporal Trajectory Analysis for Speed Skating Form Breakdown & Fatigue Prediction*

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2F1GPU-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Edge%20Optimized-green.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Formal Research Question
> **To what extent can deep learning models leverage comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation prior to observable athletic deceleration in elite speed skaters?**

---

## 💡 Why This Project Matters (Real-World Impact)
Traditional sports biomechanics relies on subjective human observation or expensive, fixed laboratory motion-capture equipment. This project builds an automated, accessible alternative targeting an unexplored niche in ice and inline speed sports:
* **Proactive Injury Prevention:** Rather than waiting for an overuse injury or a severe fall, the system learns an individual skater's normal movement baseline during fresh runs and triggers real-time warning flags when mechanical form drifts.
* **Objective Coaching Intelligence:** By comparing developing athletes against elite, world-class reference forms (such as Sven Kramer, Jorrit Bergsma, Haralds Silovs, and Mia Manganello Kilburg), the system provides concrete, data-driven feedback on kinematics rather than guessing.
* **Early-Warning Capability:** Demonstrates that deep temporal trajectory analysis can catch coordination and pacing breakdowns **long before** macroscopic deceleration physically occurs.

---

## 🛠️ System Architecture & Pipeline

```text
[Raw Long-Form Video / Local MP4] ➔ [MediaPipe PoseLandmarker] ➔ [3D Coordinate Extraction]  
                                                                         │
[UI Metrics, CSV Reports & ONNX Edge] ◄── [PyTorch / ONNX Engine] ◄── [Butterworth Filter]

## ⚡ Core Project Status & Architecture

* **Phase 1 (Completed - Proof of Concept):** Successfully proved that deep learning autoencoders and LSTM architectures can utilize comparative temporal joint-angle trajectories across segmented clips to proactively forecast biomechanical performance degradation before observable athletic deceleration occurs.
* **Phase 2 (Completed - Automated Video Ingestion, Baseline Calibration, Edge ONNX & UI Polish):** Successfully implemented an automated end-to-end video ingestion engine (`pipeline_engine.py`), automated statistical baseline calibration ($\mu + 2\sigma$), ONNX model quantization/runtime edge integration (`skating_model.onnx`), and live video auto-digestion directly into the Streamlit web dashboard (`dashboard.py`) complete with interactive metric tooltips and instant CSV report exporting.

---

## 📊 Core Pipeline & Dashboard Capabilities

* **🎥 Automated Video Ingestion Pipeline (`pipeline_engine.py`):** Automatically processes raw, unsegmented MP4 video files end-to-end to extract keypoints, compute joint angles, and run autoencoder reconstruction loss calculations.
* **⚡ Edge Model Optimization & ONNX Integration (`skating_model.onnx`):** Compiles autoencoder weights into framework-independent ONNX format, enabling fast, low-latency local edge inference directly inside the Streamlit auto-digest workflow.
* **🖥️ Interactive Web App Dashboard (`dashboard.py`):** Fully deployed Streamlit dashboard featuring an **"Auto-Digest New Video (Upload)"** mode, live time-series tracking, dynamic skater selection dropdowns, interactive anomaly threshold sliders, metric calculation cards with helpful tooltips, native YouTube video embeds for elite reference profiles, and downloadable CSV summary reports.
* **📈 Automated Baseline Calibration (`calibrate_baseline`):** Programmatically computes statistical means and standard deviations over initial window streams to establish objective, data-driven anomaly detection boundaries without manual slider guesswork.
* **🤖 AI-Powered Pose Estimation:** Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker API.
* **📐 Biomechanical Angle Calculation:** Extracts 3D spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.
* **📉 Signal Noise Reduction:** Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.
* **📊 Multi-State Kinematic Comparison:** Successfully achieved robust comparative visualization between early-stage fresh performance and late-stage fatigue trajectories, isolating true biomechanical divergence.
* **👥 Multi-Skater Pipeline Expansion:** Successfully scaled the framework beyond single-subject testing to ingest, process, and evaluate multiple elite athletes (including Mia Manganello Kilburg, Patrick Meek, Ragne Wiklund, and Carlijn Schoutens).
* **🧠 Unsupervised Deep Learning Anomaly Detection:** Utilizes deep autoencoders trained exclusively on clean, fresh baseline data to flag mechanical drift without needing pre-labeled failure sets.
* **🤖 Supervised Binary Classification (LSTM):** Implements Long Short-Term Memory (LSTM) neural networks for pre-deceleration classification and threshold optimization.
* **⏱️ Predictive Lead-Time Experimentation:** Validated tracking pipelines measuring the exact temporal window between model-flagged reconstruction error spikes and physical athletic deceleration.
* **⚡ Dynamic Statistical Thresholding ($\mu + 2\sigma$):** Automatically calculates real-time sports-science anomaly boundaries derived strictly from the skater's initial fresh baseline frames.
* **🧪 Synthetic Failure Stress-Testing (`stress_test.py`):** Validates model robustness and joint isolation via targeted perturbation analysis.
* **📦 Automated Batch Multi-File Processing (`batch_evaluate_skaters.py`):** Automatically loops through dataset files to generate consolidated executive summaries (`summary_report.csv`).

---

## 🚀 Future Development Roadmap (Advanced Extensions)

* **⛸️ Automated Stride Segmentation:** Implementing peak-detection algorithms on hip/ankle coordinates to automatically segment continuous video streams into individual stride cycles.
* **👥 User-Independent Cross-Validation:** Training models on multi-skater datasets and validating generalization performance on completely unseen athletes.
* **🔬 Sensitivity Analysis & Ablation Studies:** Performing systematic joint-removal experiments to identify which anatomical trajectories carry the highest predictive weight.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class professional speed skaters to create a gold-standard baseline folder.

---

## 📈 Project Progress & Hours Log

You can track the detailed engineering timeline, roadblocks, and solutions in our `HOURS.md` and `JOURNAL.md` files. Comprehensive capability breakdowns are available in `abilities_phase1.md` and `abilities_phase2.md`.

📂 Repository Structure

biomechanics-project/
│
├── main.py                # Core PyTorch model training & joint error decomposition
├── app.py                 # Interactive Streamlit web dashboard with dynamic thresholding
├── dashboard.py           # Polished main Streamlit dashboard with ONNX & Phase 2 export
├── pipeline_engine.py     # Automated end-to-end video processing backend
├── skating_model.onnx     # Optimized ONNX edge runtime model weights
├── data_loader.py         # Video ingestion & MediaPipe processing
├── biomechanics_utils.py  # Butterworth filter & angle math calculations
├── src/                   # Advanced processing scripts (batch eval, stress test, heatmaps)
├── Docs/                  # Visual artifacts and documentation (including joint error heatmaps)
├── data/                  # Raw and processed multivariate CSV datasets (elite speed skating trials)
├── models/                # Saved PyTorch autoencoder weights (.pth)
├── outputs/               # Consolidated multi-file batch execution telemetry & reports
├── requirements.txt       # Explicit python dependency tracking
├── HOURS.md               # Quantitative time and development log
├── JOURNAL.md             # Engineering thought process & milestones
├── abilities_phase1.md    # Phase 1 completed capabilities and milestone log
└── abilities_phase2.md    # Phase 2 active objectives and automation roadmap update
