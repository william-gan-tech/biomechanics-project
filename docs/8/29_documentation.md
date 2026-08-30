# 8/29: Comprehensive Project Documentation & Presentation Guide (`8/29_documentation.md`)

## 📋 Overview & Purpose
This document serves as the master tracking and presentation blueprint for the deep learning biomechanics engine. It synthesizes all core deliverables from Phase 1 and Phase 2, mapping out code architecture, interactive dashboard features, visual demonstration assets, and presentation slide structures.

---

## ⚡ Phase 1 & Phase 2 Summary of Uses

### Phase 1: Proof of Concept & Foundational Research
* **Core Objective:** Establish whether deep learning architectures can utilize comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation before observable athletic deceleration.
* **Key Components:**
  * **3D Spatial Mapping:** MediaPipe PoseLandmarker API extracting key skeletal coordinates ($x, y, z$).
  * **Signal Processing:** Digital Butterworth low-pass filtering to remove camera jitter and high-frequency noise.
  * **Unsupervised Autoencoders:** PyTorch-based neural networks trained exclusively on clean, fresh baseline movement data to flag mechanical drift via Reconstruction Error (Mean Squared Error).
  * **Predictive Validation:** Validating the temporal window delta between model-flagged anomaly spikes and physical velocity decay.

### Phase 2: Automation, Edge Optimization & Pipeline Scaling
* **Core Objective:** Transition from manual video segmenting to an automated, edge-optimized production system with live web telemetry and automated baseline calibration.
* **Key Components:**
  * **Automated Ingestion (`pipeline_engine.py`):** End-to-end processing of raw, unsegmented MP4 video files and YouTube streams.
  * **Baseline Calibration (`calibrate_baseline`):** Programmatic calculation of statistical anomaly boundaries ($\mu + 2\sigma$) derived strictly from initial window streams.
  * **Edge Optimization (`skating_model.onnx`):** Model quantization and ONNX runtime integration for fast local and cloud inference without heavy framework overhead.
  * **Interactive Streamlit Dashboard (`dashboard.py`):** Persistent session states (`st.session_state`), live anomaly threshold sliders, multi-axis radar profiles, and instant CSV reporting.

---

## 📸 Streamlit Dashboard Screenshot & Visual Guide

Include the following sections in your presentation or project portfolio to highlight the graphical user interface:

* **Auto-Digest Upload Panel:** Displays the drag-and-drop file uploader and YouTube URL input field for live video processing.
* **Real-Time Time-Series Trace:** Line chart tracking reconstruction error over time, featuring a dynamic threshold line ($\mu + 2\sigma$) and highlighted anomaly spikes.
* **Joint Decomposition Heatmap:** Bar charts breaking down error rates independently across knees, hips, and ankles to pinpoint specific anatomical failure points.
* **Export & Reporting Card:** Metric cards displaying peak anomaly windows alongside a one-click **"Download CSV Report"** button.

---

## 🎥 Video Demonstration Structure

When recording or embedding a video walkthrough of your project, follow this recommended 60-second script/flow:
1. **Introduction (0:00 - 0:10):** Introduce the formal research question and the real-world need for proactive injury prevention in speed skating.
2. **Live Auto-Digest Upload (0:10 - 0:30):** Upload a sample skating MP4 or paste a YouTube link into the Streamlit dashboard, showing the automated extraction and Butterworth filtering in real-time.
3. **Threshold & Anomaly Detection (0:30 - 0:45):** Adjust the dynamic anomaly threshold slider to demonstrate how statistical boundaries flag form breakdown before physical deceleration.
4. **CSV Export & Summary (0:45 - 1:00):** Download the generated telemetry CSV report and summarize the project's Phase 1 and Phase 2 milestones.

---

## 📊 Google Slides Presentation Structure

Use the following slide-by-slide outline to build your formal project presentation:

| Slide # | Slide Title | Core Content & Bullet Points |
| :--- | :--- | :--- |
| **Slide 1** | Title & Research Question | • Official Title: Predicting Fatigue-Induced Velocity Decay in Speed Skaters via Temporal Joint-Angle Kinematic Analysis<br>• Formal Research Question statement. |
| **Slide 2** | Real-World Impact & Motivation | • Limitations of subjective coaching and expensive lab motion capture.<br>• Proactive injury prevention and objective coaching intelligence. |
| **Slide 3** | Phase 1: Foundational Pipeline | • MediaPipe 3D joint coordinate tracking.<br>• Digital Butterworth low-pass filtering.<br>• Unsupervised PyTorch autoencoders and reconstruction error (MSE). |
| **Slide 4** | Phase 2: Automation & Edge Scaling | • Automated video pipeline ingestion (`pipeline_engine.py`).<br>• ONNX runtime edge optimization (`skating_model.onnx`).<br>• Statistical baseline calibration ($\mu + 2\sigma$). |
| **Slide 5** | Interactive Dashboard Architecture | • Streamlit web app interface (`dashboard.py`).<br>• Persistent session state management and multi-analysis sidebar modes.<br>• Live CSV export and professional feedback generation. |
| **Slide 6** | Results & Predictive Lead-Time | • Proof that anomaly spikes precede physical deceleration.<br>• Joint-specific error decomposition findings.<br>• Summary of validation stress tests. |
| **Slide 7** | Future Roadmap & Conclusion | • Cross-skater generalization and ablation studies.<br>• Final project takeaway and GitHub repository link. |