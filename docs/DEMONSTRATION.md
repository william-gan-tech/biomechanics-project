# ⚡ Biomechanics & Deep Learning Anomaly Detection: System Demonstration Guide (`demonstration.md`)

## 📋 Overview
This document outlines the operational execution, pipeline architecture, and cloud deployment of the advanced biomechanics anomaly detection system. Designed to analyze high-speed athletic video streams, this framework extracts 3D joint kinematics, evaluates temporal movement patterns via deep learning autoencoders, and surfaces insights through an interactive cloud-hosted web dashboard.

---

## 🛠️ System Architecture & Workflow

### 1. Data Ingestion & Pose Estimation (`src/pose_estimation.py`)
* **Input Stream:** Processes long-form video files (e.g., 6-minute time trials) or localized clip samples using OpenCV (`cv2`).
* **Skeletal Tracking:** Leverages MediaPipe's modern PoseLandmarker to track 3D body joints frame-by-frame.
* **Spatial Mapping:** Computes 3D Euclidean coordinates ($x, y, z$) to derive precise anatomical angles (knee flexion, hip angle, ankle dorsiflexion, torso lean) rather than relying on flat 2D pixel approximations.

### 2. Kinematic Processing & Window Segmentation
* **Signal Filtering:** Applies digital Butterworth low-pass filters to eliminate high-frequency camera noise and coordinate jitter.
* **Sliding Window Slicing:** Chops continuous time-series streams into overlapping temporal matrices to evaluate movement dynamics over time.
* **Differential State Slicing:** Isolates distinct comparative phases—such as early laps (Fresh State) versus final laps (Fatigued State)—to evaluate structural mechanical breakdown.

### 3. Deep Learning Anomaly Detection (PyTorch)
* **Unsupervised Reconstruction:** Trains deep learning autoencoders on fresh baseline movement data, allowing the network to learn efficient kinematic patterns without pre-labeled failure data.
* **Reconstruction Error Scoring:** Measures Mean Squared Error (MSE) across a neural bottleneck; movement patterns that deviate from efficient form produce high error spikes.
* **Multi-Joint Feature Decomposition:** Isolates loss tracking per anatomical region (e.g., tracking knee flexion independently) to pinpoint precise mechanical failure points.

---

## 🌐 Interactive Dashboard Demonstration (`src/dashboard.py`)

The pipeline outputs are visualized in real-time through an interactive web application deployed on **Streamlit Community Cloud**.

### Key Demonstration Features:
* **Multi-Analysis Sidebar Controls:** Toggle seamlessly between:
  * *Cross-Skater Anomaly & Generalization (Sven Kramer)*
  * *3000m Fresh vs. Fatigued Comparison*
  * *First-Ever Baseline Analysis*
* **Interactive Click-to-Filter Data Tables:** 
  * *Model Generalization:* Click rows to dynamically view performance metrics across source models and target skaters.
  * *Anatomical Feature Importance:* Click joint rows in the ablation table to instantly highlight and isolate that specific joint's error trace in the decomposition charts.
* **Dynamic Thresholding:** Adjust the anomaly threshold slider in real-time to observe how statistical boundaries ($\mu + 2\sigma$ or $\mu + 3\sigma$) flag structural breakdown.
* **Automated Visualizations:** Real-time generation of global reconstruction error curves, feature ablation ranking charts, and joint-specific trajectory comparisons.

---

## 🚀 Running the Demonstration Locally

To spin up the dashboard and pipeline locally on your machine, follow these steps:

1. **Clone the Repository & Navigate to Project Folder:**
   ```bash
   cd path/to/Research - biomechanics_project

Verify Dependencies:
Ensure all required packages (PyTorch, Streamlit, Pandas, OpenCV, MediaPipe, etc.) are installed via your environment:

Bash
pip install -r requirements.txt

Launch the Streamlit Dashboard:

Bash
python -m streamlit run src/dashboard.py

Access the Web Interface:
Streamlit will automatically launch a local server and open your web browser (typically at http://localhost:8501).

🏆 Project Milestone Summary
Phase 1 Validation Achieved: Successfully processed a full 6-minute speed trial, proving that deep learning autoencoders can leverage temporal joint-angle trajectories to successfully differentiate between fresh and fatigued skating mechanics prior to measurable athletic deceleration.
