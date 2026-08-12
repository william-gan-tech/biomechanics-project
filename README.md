# 🧊 Deep Learning Biomechanics & Injury Prevention Engine
*Advanced Temporal Trajectory Analysis for Speed Skating Form Breakdown & Fatigue Prediction*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red.svg)](https://pytorch.org/)
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

Impact: Ensures the model remains robust and generalizable across different subjects, preventing overfitting to a single individual's baseline mechanics.

Raw Video Ingestion: Automatically reads and processes video frames using OpenCV (cv2).

AI-Powered Pose Estimation: Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker.

Biomechanical Angle Calculation: Extracts spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.

Signal Noise Reduction: Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.

Unsupervised Deep Learning Anomaly Detection: Utilizes a Multi-Channel Autoencoder trained exclusively on clean, fresh baseline data to flag mechanical drift without needing pre-labeled failure sets.

Sliding Window Kinematic Segmentation: Chops continuous video streams into overlapping 30-frame temporal chunks to analyze movement dynamics over time.

Quantitative Fatigue Scoring: Computes Mean Squared Error (MSE) between input strides and network-reconstructed strides to output an objective Anomaly Score.

---

## 🚀 Future Development Roadmap

* **📏 Automated Data Normalization:** Spatial scaling modules to normalize joint coordinates across diverse athletes, ensuring evaluation focuses on pure form.
* **⚡ Dynamic Statistical Thresholding:** Moving to real-time statistical boundaries ($\mu + 2\sigma$) derived from initial baseline runs.
* **🔍 Joint-Specific Error Decomposition:** Upgrading the loss function to isolate reconstruction error per anatomical region (e.g., knee flexion vs. upper body posture).
* **⏱️ Quantitative Lead-Time Analysis:** Measuring the exact frame/second advantage the autoencoder provides prior to visible deceleration.
* **🖥️ Real-Time Edge UI (Streamlit / OpenCV):** Transitioning to a live dashboard capable of pulling rinkside webcam feeds and issuing instant alerts.


---

## 📈 Project Progress & Hours Log
You can track the detailed engineering timeline, roadblocks, and solutions in our HOURS.md and JOURNAL.md files. What my model in currently do is in the CAPABILITIES.md file.

## 📂 Repository Structure

```text
biomechanics-project/
│
├── main.py                 # Core orchestration script
├── data_loader.py          # Video ingestion & MediaPipe processing
├── biomechanics_utils.py   # Butterworth filter & angle math calculations
├── fatigue_results.csv     # Exported frame-by-frame anomaly metrics
├── fatigue_trend_plot.png  # Programmatic visualization artifact
├── HOURS.md                # Quantitative time and development log
├── JOURNAL.md              # Engineering thought process & milestones
└── abilities.md            # Detailed project capability breakdown

