# 🧊 Deep Learning Biomechanics & Injury Prevention Engine
*Advanced Temporal Trajectory Analysis for Speed Skating Form Breakdown & Fatigue Prediction*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2F1GPU-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Edge%20Optimized-green.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Formal Research Question
> **To what extent can deep learning models leverage comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation prior to observable athletic deceleration in elite speed skaters?**

---

## A Little Self Introduction
> **As someone deeply immersed in sports, AI, and robotics, I decided to build this model because I am a speed skater myself, having competed internationally and earned national gold medals representing Team USA while also sharing a background in endurance sports like cross country, track, and running half marathons like the San Jose and San Francisco half marathons. Between experiencing those sports firsthand and achieving a #11 world ranking in VEX robotics, I've always been fascinated by how technology and athletics intersect. For years, I watched runners and skaters struggle with fatigue, noticing that humans usually only spot form breakdown, like skaters not bending their knees or runners losing proper posture, after it has already happened and they are already slowing down. That sparked my core question: to what extent can deep learning models leverage temporal joint-angle trajectories to anticipate biomechanical performance degradation prior to measurable athletic deceleration in speed skaters? By combining my athletic background with my love for AI and robotics, I wanted to build something that moves past reactive observation into true predictive intelligence, helping skaters optimize their form and prevent injuries before fatigue even sets in.**

---


## 💡 Why This Project Matters (Real-World Impact)
Traditional sports biomechanics relies on subjective human observation or expensive, fixed laboratory motion-capture equipment. This project builds an automated, accessible alternative targeting an unexplored niche in ice and inline speed sports:
* **Proactive Injury Prevention:** Rather than waiting for an overuse injury or a severe fall, the system learns an individual skater's normal movement baseline during fresh runs and triggers real-time warning flags when mechanical form drifts.
* **Objective Coaching Intelligence:** By comparing developing athletes against elite, world-class reference forms (such as Sven Kramer, Jorrit Bergsma, Haralds Silovs, and Mia Manganello Kilburg), the system provides concrete, data-driven feedback on kinematics rather than guessing.
* **Early-Warning Capability:** Demonstrates that deep temporal trajectory analysis can catch coordination and pacing breakdowns **long before** macroscopic deceleration physically occurs.

---

## ⚡ Core Project Status & Architecture

* **Phase 1 (Completed - Proof of Concept):** Successfully proved that deep learning autoencoders and LSTM architectures can utilize comparative temporal joint-angle trajectories across segmented clips to proactively forecast biomechanical performance degradation before observable athletic deceleration occurs.
* **Phase 2 (Completed - Automated Video Ingestion, Baseline Calibration, Edge ONNX & UI Polish):** Fully finalized end-to-end video pipeline automation (`src/pipeline_engine.py`), automated statistical baseline calibration ($\mu + 2\sigma$), ONNX model quantization/runtime edge integration (`skating_model.onnx`), persistent Streamlit session state management, and live video auto-digestion directly into the web dashboard (`src/dashboard.py`).

---

## 📊 Core Pipeline & Dashboard Capabilities

* **🎥 Automated Video Ingestion Pipeline (`src/pipeline_engine.py`):** Automatically processes raw, unsegmented MP4 video files end-to-end to extract keypoints, compute joint angles, and run autoencoder reconstruction loss calculations.
* **⚡ Edge Model Optimization & ONNX Integration (`skating_model.onnx`):** Compiles autoencoder weights into framework-independent ONNX format, enabling fast, low-latency local edge inference directly inside the Streamlit auto-digest workflow.
* **🖥️ Interactive Web App Dashboard (`src/dashboard.py`):** Fully deployed Streamlit dashboard featuring persistent session states (`st.session_state`), an **"Auto-Digest New Video (Upload)"** mode, live time-series tracking, dynamic skater selection dropdowns, interactive anomaly threshold sliders ($0.70$ to $0.99$ multipliers), metric calculation cards with helpful tooltips, native YouTube video embeds for elite reference profiles, and downloadable CSV summary reports.
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

## 🚀 Getting Started & Execution

To avoid relative import pathing issues and ensure absolute path safety across environments, execute the application and scripts using Python's module (`-m`) flag:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the primary Streamlit dashboard
python -m streamlit run src/dashboard.py

# Run standalone pipeline ingestion engine
python -m src.pipeline_engine

```

## 🔮 Future Development Roadmap (Advanced Extensions)
* **👥 User-Independent Cross-Validation:** Training models on multi-skater datasets and validating generalization performance on completely unseen athletes.
* **🔬 Sensitivity Analysis & Ablation Studies:** Performing systematic joint-removal experiments to identify which anatomical trajectories carry the highest predictive weight.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class professional speed skaters to create a gold-standard baseline folder.

---


## 📈 Project Progress & Hours Log

Detailed engineering timelines, challenges (such as module resolution optimization and third-party streaming constraints), and technical solutions are tracked in `HOURS.md` and `JOURNAL.md`. Comprehensive capability logs are available in `CAPABILITIES_PHASE1.md` and `CAPABILITIES_PHASE2.md`. Additional repository tracking files include `DEMONSTRATION.md`, `MILESTONES.md`, `README.md`, `REPOSITORYSTRUCTURE.md`, `REQUIREMENTS.MD`, and `STREAMLITVERSIONS.md`.

* **Recent Documentation Updates:** The newly added `8-30-2026_documentation.md` (committed 1 minute ago) revises documentation dates and enhances overall content structure.
* **Phase & Milestone Tracking:** `CAPABILITIES_PHASE1.md`, `CAPABILITIES_PHASE2.md`, and `MILESTONES.md` log core capability frameworks, phase roadmaps, and achievement logs.
* **Engineering Logs & Dependencies:** `HOURS.md` and `JOURNAL.md` record daily engineering timelines and technical problem-solving, supported by `REQUIREMENTS.MD` (updating yt-dlp version details) and `STREAMLITVERSIONS.md` (tracking Streamlit framework versions).
---

## 🛠️ System Architecture & Pipeline
```text
[Raw Long-Form Video / Local MP4] ➔ [MediaPipe PoseLandmarker] ➔ [3D Coordinate Extraction]  
                                                                        │
[UI Metrics, CSV Reports & ONNX Edge] ◄── [PyTorch / ONNX Engine] ◄── [Butterworth Filter]
