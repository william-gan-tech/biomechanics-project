# 🧊 Deep Learning Biomechanics & Injury Prevention Engine
*Advanced Temporal Trajectory Analysis for Speed Skating Form Breakdown & Fatigue Prediction*

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2F1GPU-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Formal Research Question
> **To what extent can deep learning models leverage temporal joint-angle trajectories to anticipate biomechanical performance degradation prior to measurable athletic deceleration in speed skaters?**

---

## 💡 Why This Project Matters (Real-World Impact)
Traditional sports biomechanics relies on subjective human observation or expensive, fixed laboratory motion-capture equipment. This project builds an automated, accessible alternative targeting an unexplored niche in ice and inline speed sports:
* **Proactive Injury Prevention:** Rather than waiting for an overuse injury or a severe fall, the system learns an individual skater's normal movement baseline during fresh runs and triggers real-time warning flags when mechanical form drifts.
* **Objective Coaching Intelligence:** By comparing developing athletes against elite, world-class reference forms, the system provides concrete, data-driven feedback on kinematics rather than guessing.
* **Early-Warning Capability:** Demonstrates that deep temporal trajectory analysis can catch coordination and pacing breakdowns **long before** macroscopic deceleration physically occurs.

---

## 🛠️ System Architecture & Pipeline

[Raw Long-Form Video] ➔ [MediaPipe PoseLandmarker] ➔ [3D Coordinate Extraction]  
                                                                            │
[Summary Reports & Multi-Panel Plots] ➔ [PyTorch Autoencoder MSE] ➔ [Butterworth Low-Pass Filter]

---

## ⚡ Core Pipeline & Dashboard Capabilities (Fully Implemented)

* **🎥 Long-Form Video Ingestion:** Automatically reads and processes lengthy performance streams (such as 6-minute time trials via OpenCV `cv2`).
* **🤖 AI-Powered Pose Estimation:** Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker API.
* **📐 Biomechanical Angle Calculation:** Extracts 3D spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.
* **📉 Signal Noise Reduction:** Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.
* **🗺️ Precision Frame-Range Preprocessing:** Configures exact boundary mapping to extract clean comparative subsets for fresh vs. fatigued performance states.
* **📊 Multi-State Kinematic Comparison:** Successfully achieved robust comparative visualization between early-stage fresh performance and late-stage fatigue trajectories, isolating true biomechanical divergence.
* **👥 Multi-Skater Pipeline Expansion:** Successfully scaled the framework beyond single-subject testing to ingest, process, and evaluate multiple elite athletes (including Mia Manganello Kilburg and Patrick Meek).
* **🧠 Unsupervised Deep Learning Anomaly Detection:** Utilizes deep autoencoders trained exclusively on clean, fresh baseline data to flag mechanical drift without needing pre-labeled failure sets.
* **🤖 Supervised Binary Classification (LSTM):** Implements Long Short-Term Memory (LSTM) neural networks for pre-deceleration classification and threshold optimization.
* **⏳ Comparative Temporal Depth Architectures:** Benchmarked Feed-Forward Autoencoders, Long Short-Term Memory (LSTM) Autoencoders, and Temporal Convolutional Networks (TCNs) to evaluate sequence depth and long-term dependency modeling.
* **⏱️ Predictive Lead-Time Experimentation:** Validated tracking pipelines measuring the exact temporal window between model-flagged reconstruction error spikes and physical athletic deceleration.
* **🎞️ Sliding Window Kinematic Segmentation:** Chops continuous video streams into overlapping 30-frame temporal chunks to analyze movement dynamics over time rather than isolated frames.
* **🔍 Joint-Specific Reconstruction Error Decomposition:** Isolates reconstruction error independently across anatomical regions (Left/Right Knees, Left/Right Hips) to pinpoint precise failure points.
* **⚡ Dynamic Statistical Thresholding ($\mu + 2\sigma$ / $\mu + 3\sigma$):** Automatically calculates real-time sports-science anomaly boundaries derived strictly from the skater's initial fresh baseline frames.
* **🧪 Synthetic Failure Stress-Testing (`stress_test.py`):** Validates model robustness and joint isolation via targeted perturbation analysis.
* **📦 Automated Batch Multi-File Processing (`batch_evaluate_skaters.py`):** Automatically loops through dataset files (`mia_fresh.csv`, `mia_fatigued.csv`, etc.) to generate consolidated executive summaries (`summary_report.csv`).
* **🎨 Visual Time-Series Heatmaps & Research Plots:** Produces detailed multi-joint error intensity timelines and absolute frame-mapped comparative research visualizations (`compare_research.py`).
* **🌐 Interactive Web Dashboard (`app.py`):** Fully deployed Streamlit dashboard featuring live time-series tracking, dynamic skater selection dropdowns (supporting Mia Manganello Kilburg, Patrick Meek, and Sven Kramer), interactive anomaly threshold sliders, multi-joint selection dropdowns, metric calculation cards, and downloadable CSV summary reports.
* **🛠️ Local Multi-Subject Architecture & Git Integration:** Clean directory management (`data/`, `models/`, `outputs/`, `Docs/`, `src/`) with full version control tracking via GitHub.

---

## 🚀 Future Development Roadmap

* **📊 Advanced Temporal Window Refactoring:** Upgrading comparative research visualization logic to support multi-segment analysis across diverse athletic trials.
* **⛸️ Automated Stride Segmentation:** Implementing peak-detection algorithms on hip/ankle coordinates to segment video streams into individual stride cycles.
* **👥 User-Independent Cross-Validation:** Training models on multi-skater datasets and validating generalization performance on completely unseen athletes.
* **🔬 Sensitivity Analysis & Ablation Studies:** Performing systematic joint-removal experiments to identify which anatomical trajectories carry the highest predictive weight.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class professional speed skaters to create a gold-standard baseline folder.

---

## 📈 Project Progress & Hours Log
You can track the detailed engineering timeline, roadblocks, and solutions in our `HOURS.md` and `JOURNAL.md` files. A comprehensive capability breakdown is available in `abilities.md`.

---

## 📂 Repository Structure

```text
biomechanics-project/
│
├── main.py                # Core PyTorch model training & joint error decomposition
├── app.py                 # Interactive Streamlit web dashboard with dynamic thresholding
├── data_loader.py         # Video ingestion & MediaPipe processing
├── biomechanics_utils.py  # Butterworth filter & angle math calculations
├── src/                   # Advanced processing scripts (batch evaluation, stress test, heatmaps, preprocessing)
├── Docs/                  # Visual artifacts and documentation (including joint error heatmaps)
├── data/                  # Raw and processed multivariate CSV datasets (Mia Manganello Kilburg, Patrick Meek, etc.)
├── models/                # Saved PyTorch autoencoder weights (.pth)
├── outputs/               # Consolidated multi-file batch execution telemetry & reports
├── requirements.txt       # Explicit python dependency tracking
├── HOURS.md               # Quantitative time and development log
├── JOURNAL.md             # Engineering thought process & milestones
└── abilities.md           # Detailed project capability breakdown
