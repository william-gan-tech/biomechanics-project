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
* **Phase 3 (In Progress - Multi-Angle Stream Fusion & Cross-Athlete Generalization):** Developing synchronized multi-camera ingestion (`multi_view_fusion.py`) with anchor-point spatial alignment, cross-subject bone scaling matrices, and asynchronous multi-threaded queueing.

---

## 📊 Core Pipeline & Dashboard Capabilities

* **🌐 Multi-Angle Camera Stream Fusion:** Ingests synchronized multi-angle video inputs (e.g., lateral profile tracking vs. head-on view) to eliminate blind spots using anchor-point spatial alignment, Dynamic Time Warping (DTW), and high-precision timestamp interpolation.
* **👥 Generalized Cross-Subject Bone Scaling:** Integrates dynamic normalization matrices into feature extraction to adapt across diverse body types and limb lengths, reducing reconstruction MSE from ~4,500 down to ~0.62.
* **⚡ ONNX Edge Runtime Acceleration:** Compiles PyTorch LSTM autoencoder weights into optimized ONNX format (`skating_model.onnx`) with explicit dynamic axes for batch size and sequence length to lower CPU/GPU latency.
* **⚡ Asynchronous Multi-Threaded Queueing:** Upgrades processing pipelines and `yt_dlp` wrapper utilities with threaded chunked downloading and parallel frame extraction to prevent Streamlit UI thread blocking.
* **🎥 Automated Video Ingestion Pipeline (`src/pipeline_engine.py`):** Automatically processes raw, unsegmented MP4 video files and live remote URLs end-to-end to extract keypoints, compute joint angles, and run autoencoder reconstruction loss calculations.
* **🖥️ Interactive Web App Dashboard (`src/dashboard.py`):** Fully deployed Streamlit dashboard featuring persistent session states (`st.session_state`), an **"Auto-Digest New Video"** mode, live time-series tracking, dynamic skater selection dropdowns, interactive anomaly threshold multipliers, metric calculation cards, native YouTube video embeds, and downloadable CSV summary reports.
* **🎬 Annotated Video Rendering & Export Engine:** Processes live video streams to output downloadable `.mp4` visualization files complete with real-time skeleton point overlays, HUD telemetry, and active bone-length scaling lines.
* **📈 Automated Baseline Calibration (`calibrate_baseline`):** Programmatically computes statistical means and standard deviations over initial window streams to establish objective, data-driven anomaly detection boundaries.
* **🤖 AI-Powered Pose Estimation:** Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker API.
* **📐 Biomechanical Angle Calculation:** Extracts 3D spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.
* **📉 Signal Noise Reduction:** Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.
* **📊 Multi-State Kinematic Comparison:** Successfully achieves robust comparative visualization between early-stage fresh performance and late-stage fatigue trajectories.
* **🧠 Unsupervised Deep Learning Anomaly Detection:** Utilizes deep autoencoders trained exclusively on clean, fresh baseline data to flag mechanical drift without needing pre-labeled failure sets.
* **🤖 Supervised Binary Classification (LSTM):** Implements Long Short-Term Memory neural networks for pre-deceleration classification and threshold optimization.
* **⏱️ Predictive Lead-Time Experimentation:** Validates tracking pipelines measuring the exact temporal window between model-flagged reconstruction error spikes and physical athletic deceleration.
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

* **Multi-Athlete Concurrent Tracking:** Expand the multi-view fusion engine to simultaneously segment and track multiple interacting athletes on an oval or rink in real-time.
* **Hardware-Accelerated Edge Deployment:** Port the ONNX runtime model to dedicated embedded edge hardware (such as NVIDIA Jetson or Raspberry Pi with Coral TPU accelerators) for courtside coaching feedback.
* **Automated Kinetic Chain Correction:** Integrate reinforcement learning feedback loops to automatically suggest real-time physical adjustments when anomalous joint trajectories are flagged.
* **Expanded Cross-Sport Generalization:** Train and validate the autoencoder architecture across additional continuous-motion sports (e.g., speed skating to cycling and rowing) to test transfer learning performance.
* **Robust Temporal-Smoothing & Confidence Gating:** Implement Exponential Moving Average (EMA) coordinate filtering and confidence-score threshold masking to completely resolve high-speed landmark jitter and anchor-point misalignments during dynamic athletic leans.
---

## 📈 Project Progress & Hours Log

* **Engineering Timelines & Troubleshooting:** Detailed engineering timelines, technical challenges (such as module resolution optimization and third-party streaming constraints), and resolutions are documented in `HOURS.md` and `JOURNAL.md`.
* **Capability Frameworks:** Comprehensive capability logs are maintained in `CAPABILITIES_PHASE1.md`, `CAPABILITIES_PHASE2.md`, and `CABILITIES_PHASE3.md`.
* **Repository Tracking Files:** Additional repository tracking files include `DEMONSTRATION.md`, `MILESTONES.md`, `README.md`, `REPOSITORYSTRUCTURE.md`, `REQUIREMENTS.md`, and `STREAMLITVERSIONS.md`.
* **Recent Documentation Updates:** Documentation syncs incorporate Phase 3 multi-view fusion architecture and cross-athlete validation frameworks.

---

## 🛠️ System Architecture & Pipeline

```text
[Raw Multi-Angle Video Streams] ➔ [Multi-View Fusion & Spatial Alignment] ➔ [MediaPipe PoseLandmarker]  
                                                                                    │
[UI Metrics, CSV Reports & ONNX Edge] ◄── [PyTorch / ONNX Engine] ◄── [Butterworth & Normalization Filter]
