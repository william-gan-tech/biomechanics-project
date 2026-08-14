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

## 🛠️ Current System Architecture & Pipeline

[Raw Video (OpenCV)] ➔ [MediaPipe PoseLandmarker] ➔ [3D Coordinate Extraction]  
                                                       │
[Anomaly Score CSV & Plots] ➔ [PyTorch Autoencoder MSE] ➔ [Butterworth Low-Pass Filter]

---

## ⚡ Core Pipeline & Dashboard Capabilities (Fully Implemented)

* **🎥 Raw Video Ingestion:** Automatically reads and processes video streams using OpenCV (`cv2`).
* **🤖 AI-Powered Pose Estimation:** Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker.
* **📐 Biomechanical Angle Calculation:** Extracts 3D spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.
* **📉 Signal Noise Reduction:** Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.
* **🧠 Unsupervised Deep Learning Anomaly Detection:** Utilizes a Multi-Channel Autoencoder trained exclusively on clean, fresh baseline data to flag mechanical drift without needing pre-labeled failure sets.
* **🎞️ Sliding Window Kinematic Segmentation:** Chops continuous video streams into overlapping 30-frame temporal chunks to analyze movement dynamics over time rather than isolated frames.
* **🔍 Joint-Specific Reconstruction Error Decomposition:** Isolates reconstruction error independently across anatomical regions (Left/Right Knees, Left/Right Hips) to pinpoint precise failure points.
* **⚡ Dynamic Statistical Thresholding ($\mu + 2\sigma$):** Automatically calculates real-time sports-science anomaly boundaries derived strictly from the skater's initial fresh baseline frames.
* **🌐 Interactive Web Dashboard (`app.py`):** Fully deployed Streamlit dashboard featuring live time-series tracking, interactive anomaly threshold sliders, multi-joint selection dropdowns, metric calculation cards, and downloadable CSV summary reports.
* **🛠️ Modular Local Architecture & Git Integration:** Clean three-folder directory management (`data/`, `models/`, `outputs/`) with full version control tracking via GitHub.

---

## 🚀 Future Development Roadmap

* **📏 Automated Data Normalization:** Spatial scaling modules to normalize joint coordinates across diverse athletes, ensuring evaluation focuses on pure form.
* **⏱️ Quantitative Lead-Time Analysis:** Developing an automated evaluation script to calculate the exact frame/second advantage the autoencoder provides prior to visible deceleration.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class, professional speed skaters through the feature extractor to create a "gold-standard" baseline folder.
* **🖥️ Real-Time Edge UI Integration:** Transitioning to a live dashboard capable of pulling rinkside webcam feeds and issuing instant alerts.

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
├── biomechanics_utils.py    # Butterworth filter & angle math calculations
├── fatigue_results.csv      # Exported frame-by-frame multi-joint anomaly metrics
├── requirements.txt       # Explicit python dependency tracking
├── HOURS.md               # Quantitative time and development log
├── JOURNAL.md             # Engineering thought process & milestones
└── abilities.md           # Detailed project capability breakdown
