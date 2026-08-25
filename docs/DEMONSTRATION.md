# ⚡ Biomechanics & Deep Learning Anomaly Detection: System Demonstration Guide (`demonstration.md`)

## 📋 Overview
This document outlines the operational execution, pipeline architecture, and cloud deployment of the advanced biomechanics anomaly detection system. Designed to analyze high-speed athletic video streams, this framework extracts 3D joint kinematics, evaluates temporal movement patterns via deep learning autoencoders, and surfaces insights through an interactive cloud-hosted web dashboard.

---
## Part 1: Initial Environment Setup & Prerequisites

### Step 1: Install Python 3.12
1. Go to python.org and download Python 3.12.
2. Run the installer `.exe` file.
3. ⚠️ **CRITICAL:** Check the box at the bottom that says **"Add python.exe to PATH"** before clicking **Install Now**.

### Step 2: Install Microsoft C++ Runtime
This installs missing system files like `c10.dll` so PyTorch loads properly without crashing.
1. Download the Microsoft Visual C++ Redistributable (x64) installer from Microsoft: [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Run `vc_redist.x64.exe` and follow the prompts. Restart your computer if Windows asks you to.

### Step 3: Set Up Project Folder & Terminal
1. Create a folder on your Desktop named `Research - biomechanics_project`.
2. Open Command Prompt (Press `Win + R`, type `cmd`, and hit Enter).
3. Navigate into your folder:
   ```dos
   cd "C:\Users\<YourUsername>\OneDrive\Desktop\Research - biomechanics_project"

   Automated Setup & Demonstration
Copy and paste the following single command into your terminal to clone, install, generate data, and run the complete model demo instantly:

git clone [https://github.com/william-gan-tech/biomechanics-project.git](https://github.com/william-gan-tech/biomechanics-project.git) && cd biomechanics-project && pip install -r requirements.txt && python generate_mock_data.py && python demo_skater_a.py

(Starting from git clone and ending at demo_skater_a.py is the code you need to paste!)

After running the command above, check the outputs/demo_result.png file to view the generated model reconstruction error and fatigue threshold profile.

## 📊 Advanced Analytical & Interactive Features Included:
* **Predictive Lead-Time Experimentation:** Evaluates the temporal gap between model-flagged anomaly spikes and physical deceleration to prove early breakdown anticipation.
* **Dynamic Statistical Thresholding ($\mu + 2\sigma$):** Automatically calculates sports-science boundary limits derived strictly from the skater's initial fresh baseline frames.
* **Joint-Specific Reconstruction Error Decomposition:** Isolates reconstruction error per anatomical region independently (tracking knees vs. hips) to pinpoint exact failure points.
* **Interactive Streamlit Dashboard (`dashboard.py`):** Provides a live web interface featuring dynamic threshold sliders, real-time metric cards, time-series anomaly trend lines, multi-format YouTube/MP4 video auto-digestion, and one-click CSV report exports.

---

## 🌐 Launching the Interactive Streamlit Web Dashboard
To explore the pipeline interactively through a graphical user interface with dynamic threshold sliders and real-time metric cards:
1. Ensure your dependencies are installed (`pip install -r requirements.txt`).
2. Run the application from your terminal:
   ```bash
   python -m streamlit run src/dashboard.py

## 🛠️ System Architecture & Workflow

### 1. Data Ingestion & Pose Estimation (`src/pipeline_engine.py` & `data_loader.py`)
* **Input Stream:** Processes long-form video files (e.g., 6-minute time trials), localized MP4 uploads, or raw video streams using OpenCV (`cv2`).
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
  * *Auto-Digest New Video (Upload MP4 or Link)*
  * *Cross-Skater Anomaly & Generalization (Sven Kramer, Jorrit Bergsma, Haralds Silovs)*
  * *3000m Fresh vs. Fatigued Comparison*
  * *First-Ever Baseline Analysis*
* **Interactive Click-to-Filter Data Tables:** 
  * *Model Generalization:* Click rows to dynamically view performance metrics across source models and target skaters.
  * *Anatomical Feature Importance:* Click joint rows in the ablation table to instantly highlight and isolate that specific joint's error trace in the decomposition charts.
* **Dynamic Thresholding:** Adjust the anomaly threshold slider in real-time to observe how statistical boundaries ($\mu + 2\sigma$ or $\mu + 3\sigma$) flag structural breakdown.
* **Automated Visualizations:** Real-time generation of global reconstruction error curves, feature ablation ranking charts, and joint-specific trajectory comparisons.

## 🚀 Running the Demonstration Locally

To spin up the dashboard and pipeline locally on your machine, follow these steps:

1. **Clone the Repository & Navigate to Project Folder:**
   ```bash
   cd "C:\Users\<YourUsername>\OneDrive\Desktop\Research - biomechanics_project"

   Verify Dependencies:
Ensure all required packages (PyTorch, Streamlit, Pandas, OpenCV, MediaPipe, yt-dlp, etc.) are installed via your environment:

Bash
pip install -r requirements.txt
Launch the Streamlit Dashboard:

Bash
python -m streamlit run src/dashboard.py
Access the Web Interface:
Streamlit will automatically launch a local server and open your web browser (typically at http://localhost:8501).

🏆 Project Milestone Summary
Phase 1 & Phase 2 Validation Achieved: Successfully processed full-length skating trials and integrated automated video ingestion pipelines, proving that deep learning autoencoders can leverage temporal joint-angle trajectories to successfully differentiate between fresh and fatigued skating mechanics prior to measurable athletic deceleration.
