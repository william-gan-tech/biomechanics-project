# 8/30: Comprehensive Project Documentation & Presentation Guide 

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

### Choose different analysis modes (of our 5 currently available) at the life sidebar 

<img width="325" height="582" alt="Screenshot 2026-08-29 235138123" src="https://github.com/user-attachments/assets/38cd6df0-75d0-4c0a-9266-e53c09ea8b0a" />


### 📸 Streamlit Dashboard Telemetry & Visual Analysis (Mode 1: Cross-Skater Generalization)

### 1. Cross-Skater Generalization & Telemetry Overview
* **Model Transferability:** Evaluates how well a reference model trained on elite skater Sven Kramer generalizes to another professional athlete, Mia Manganello Kilburg.
* **Key Performance Metrics:** Demonstrates a **91.4% Cross-Subject Accuracy** with an **Optimal Transfer** status, showing that foundational elite biomechanical movement patterns share consistent structural trajectories.
* **Live Status Cards:** Displays real-time telemetry including a Mean Reconstruction Error of `0.0277`, a Peak Reconstruction Error of `0.0376`, a dynamic anomaly threshold of `0.0450`, and a **Normal Form** kinematic status confirming stable movement without acute fatigue breakdown.

---

* **Important!** The reference Model Source and Target Skater to Evaluate can be changed based on the available world-class skaters in the dataset currently, however, you can't compare a skater to himself/herself!

 <img width="1521" height="651" alt="Screenshot 2026-08-29 233739" src="https://github.com/user-attachments/assets/522fc058-6895-4b6f-a8f7-6f73855c9e6c" />
(use inspect or close up if you need a better or more clear view of the data)


### 2. Multi-Joint Decomposition Error Profile
* **Anatomical Isolation:** Plots Mean Squared Error (MSE) independently across specific body regions, including **Knee Flexion**, **Hip Angle**, **Ankle Dorsiflexion**, and **Torso Lean**across 50 consecutive time steps.
* **Threshold Monitoring:** Compares individual joint error trajectories against the red dashed **Anomaly Threshold** line ($\mu + 2\sigma$), allowing researchers to pinpoint precisely which joint or segment exhibits mechanical drift during the trial.
  
<img width="1526" height="829" alt="Screenshot 2026-08-29 233751" src="https://github.com/user-attachments/assets/7239e79c-c8ff-4de2-b74f-cf4b0e72e0e7" />

--- 

### 📸 Streamlit Dashboard Telemetry & Visual Analysis (Mode 2: 3000m Fresh vs. Fatigued Comparison)

### 1. 3000m Endurance Analysis Telemetry
* **Endurance Breakdown:** Compares early-lap (fresh) versus late-lap (fatigued) kinematic performance for elite skater Mia Manganello Kilburg.
* **Key Performance Metrics:** Displays a **Fresh Stride Frequency of 1.35 Hz** dropping to a **Fatigued Stride Frequency of 1.13 Hz** (a $-15.4\%$ velocity/frequency decay), alongside a **Reconstruction Error Delta of $0.04$** ($+72.4\%$).
* **Early Fatigue Detection:** Highlights an **Early Fatigue Detection lead-time of $1.2$ seconds**, proving the deep learning model flags mechanical form breakdown before physical deceleration fully manifests.

<img width="1504" height="407" alt="Screenshot 2026-08-29 234734" src="https://github.com/user-attachments/assets/348bafd5-6407-4f49-b92d-11c49eda0350" />

### 2. Kinematic Trajectory & MSE Progression Comparison
* **Knee Extension Angle Trajectory:** Compares Fresh State (Lap 1) solid green curves against Fatigued State (Lap 7) dashed red curves, illustrating flattening knee-extension cycles and structural breakdown under high endurance load.
* **Reconstruction Error (MSE) Progression:** Plots the clear divergence between stable low-loss fresh reconstruction error lines and surging fatigued reconstruction error peaks crossing the orange **Fatigue Alert Line** ($0.045$).

<img width="1482" height="727" alt="Screenshot 2026-08-29 234755" src="https://github.com/user-attachments/assets/a3b3d852-ee67-4680-ac60-6eeeeb75c06e" />

* **Important!** You CAN change the skater to analyze! Go to the left sidebar under analysis to change it:

<img width="312" height="582" alt="Screenshot 2026-08-29 235138" src="https://github.com/user-attachments/assets/b519eaac-459d-4bd2-8b65-6175de90ffa0" />

---

## 📸 Streamlit Dashboard Telemetry & Visual Analysis (Mode 3: Form & Technique Baseline Profile)

### 1. Form & Technique Reference Profile Metrics
* **Core Stability & Symmetry Index:** Highlights elite reference performance metrics for Sven Kramer, displaying a **98.4% Core Stability Index**, **High Reference Technique Consistency**, and a **99.1% Push-Off Symmetry** score.
* **Optimal Posture & Deviation:** Measures an **Optimal Lean Angle of 42.1°**, an extremely low **Form Deviation Score of 0.012 (Minimal)**, and verifies **Optimal (High-FPS)** baseline data quality.

<img width="1494" height="511" alt="Screenshot 2026-08-29 235351" src="https://github.com/user-attachments/assets/b0bbd1ee-ee02-4aa0-9ecb-d3643f1ea1d2" />


### 2. Baseline Kinematic Ranges Table
* **Anatomical Boundary Mapping:** Programmatically tracks precise movement boundaries across major joints under ideal conditions.
* **Key Segment Statistics:** 
  * **Knee Joint:** Min angle $35.2000^\circ$, Max angle $112.4000^\circ$, Mean Velocity $145.2000^\circ/\text{s}$.
  * **Hip Joint:** Min angle $22.1000^\circ$, Max angle $78.6000^\circ$, Mean Velocity $98.4000^\circ/\text{s}$.
  * **Ankle Dorsiflexion:** Min angle $12.4000^\circ$, Max angle $38.9000^\circ$, Mean Velocity $62.1000^\circ/\text{s}$.
  * **Torso Lean Angle:** Min angle $18.5000^\circ$, Max angle $44.2000^\circ$, Mean Velocity $24.8000^\circ/\text{s}$.

<img width="1484" height="370" alt="Screenshot 2026-08-29 235454" src="https://github.com/user-attachments/assets/d2c003eb-0f1e-4443-868a-2b0ed63c6310" />

### 3. Reference Knee & Posture Cycle Graph
* **Single-Stride Normalization:** Plots the normalized single-stride cycle pattern tracking right knee joint flexion angle across sequential video frames.
* **Mechanical Efficiency Insight:** Visualizes periodic sinusoidal waveforms representing rhythmic loading, apex extension, and recovery phases characteristic of elite mechanics, confirming optimal mechanical efficiency and minimal energy loss.

<img width="1481" height="850" alt="Screenshot 2026-08-29 235444" src="https://github.com/user-attachments/assets/b51da567-2a41-4c6a-b6d9-2844fe6382b1" />

* **Important!** You CAN change the skater to analyze! Go to the left sidebar under analysis to change it:

<img width="320" height="577" alt="Screenshot 2026-08-29 235509" src="https://github.com/user-attachments/assets/6f2bf5f9-59b1-43c5-9b56-18870b97c854" />

---

## 📸 Streamlit Dashboard Telemetry & Visual Analysis (Mode 4: First-Ever Baseline Analysis)

### 1. First-Ever Baseline Analysis Telemetry Table
* **Initial Dataset Acquisition:** Captures foundational calibration metrics for the reference subject (Sven Kramer) under ideal execution conditions.
* **Key Calibration Parameters:** Displays an **Initial Range of Motion of $91.0^\circ$** (Calibrated), a **Symmetry Index of $99.3\%$** (Verified), a low **Baseline Mean Squared Error of $0.015$** (Optimal), a **Data Capture Frequency of $120\text{ Hz}$** (Active), and a **Sensor Alignment Score of $98.8\%$** (Passed).

<img width="1358" height="472" alt="Screenshot 2026-08-29 235858" src="https://github.com/user-attachments/assets/264cfbc2-6ab4-4613-95c3-65ec6688b06c" />

### 2. Baseline Signal Calibration Plot
* **Raw Calibration Stream:** Plots the continuous sinusoidal signal output (normalized) over a 10-second calibration window for the reference subject.
* **Signal Stability:** Visualizes clean, periodic waveform oscillations representing uniform motion tracking, which acts as the foundational standard required for unsupervised autoencoder training and statistical threshold derivation ($\mu + 2\sigma$).

<img width="1476" height="764" alt="Screenshot 2026-08-29 235908" src="https://github.com/user-attachments/assets/0674bac8-80db-4449-837f-2dcd12299ed8" />

* **Important!** You CAN change the skater to analyze! Go to the left sidebar under analysis to change it:

<img width="324" height="594" alt="Screenshot 2026-08-29 235921" src="https://github.com/user-attachments/assets/8be6b1f1-c06d-4c87-bd67-c9a9c4ee48ec" />

---

## 📸 Streamlit Dashboard Telemetry & Visual Analysis (Mode 5: Pipeline & Analysis Controls Sidebar)

The control panel shown in the image represents the **Pipeline & Analysis Controls** sidebar from your Streamlit dashboard (`dashboard.py`), serving as the interactive command center for managing video ingestion and model telemetry settings. 

### Core Components & Functionality
* **Select Analysis Mode:** Allows you to toggle between operational views, specifically routing the application into **Auto-Digest New Video (Upload / Link)** processing.
* **Anomaly Threshold Slider:** Adjusts the real-time sports-science threshold boundary (set to `0.04`, ranging from `0.01` to `0.10`), which determines the exact sensitivity limit used by the deep learning autoencoder to flag mechanical form breakdown and reconstruction error spikes.
* **Kinematic Smoothing Window Slider:** Configures the digital filtering window size (set to `5`, ranging from `1` to `15`) to eliminate high-frequency camera noise and coordinate jitter from raw video frames.
* **MSE Decomposition Level Dropdown:** Controls anatomical specificity by switching loss tracking between specific body segments (such as knees or hips) and the **Full Body Aggregate**.
* **Run Full Fatigue Pipeline Button:** Acts as the primary execution trigger that initiates end-to-end video ingestion, MediaPipe 3D pose extraction, Butterworth filtering, and ONNX edge inference directly from the user interface.

<img width="282" height="551" alt="Screenshot 2026-08-30 000246" src="https://github.com/user-attachments/assets/d10bf808-1ee2-49bb-b686-710ffc442d52" />

<img width="1575" height="501" alt="Screenshot 2026-08-30 233534" src="https://github.com/user-attachments/assets/f7ff00de-36d2-45d1-ba85-5f3f1a04fc3e" />


### 1. Automated Video Ingestion & Execution Panel
* **Input Method Selection:** Configures the ingestion channel to handle either local MP4 file uploads or direct YouTube URL linking (e.g., testing with Lee Sang-Hwa).
* **Pipeline Execution Trigger:** Activates the **Download & Run Full Auto-Digest Pipeline & ONNX Inference (URL)** button to pull remote video streams, extract 3D pose landmarks via MediaPipe, and execute lightweight edge inference.
* **Execution Status Validation:** Renders success feedback (`Pipeline executed successfully!`) confirming that frames have been fully processed and passed down to the analytics engine.

<img width="1487" height="660" alt="Screenshot 2026-08-30 231758" src="https://github.com/user-attachments/assets/10d8fd2c-1221-424c-81fe-610e7cb317bb" />

### 2. Edge Device Optimization & Sensitivity Telemetry
* **ONNX Runtime Integration:** Successfully loads the framework-independent model (`skating_model.onnx`) into the ONNX Runtime engine for high-efficiency edge execution.
* **Local Processing Confirmation:** Verifies that the linked or uploaded video was completely digested using the optimized edge architecture.
* **Fatigue Detection Sensitivity Controls:** Utilizes an interactive **Threshold Peak Multiplier** slider (set to `0.92`) to dynamically scale the anomaly cutoff boundary.
* **Live Summary Metric Cards:** 
  * **Mean Loss:** Displays the aggregate reconstruction mean squared error (`0.0639`).
  * **Dynamic Threshold:** Sets the active anomaly cutoff line (`0.0736`).
  * **First Fatigue Onset:** Pinpoints the exact timestamp where the model first detected mechanical form breakdown (`4.5s`).
  * **Fatigue Time %:** Calculates the proportion of the trial spent in an anomalous, fatigued state (`24.8%`).

<img width="1489" height="489" alt="Screenshot 2026-08-30 231810" src="https://github.com/user-attachments/assets/802e5e61-8fad-4a07-be33-35fec6de5b92" />

### 1. Real-Time Reconstruction Loss & Fatigue Spikes Chart
* **Reconstruction Trajectory Mapping:** Plots raw reconstruction MSE loss (light grey) alongside a 30-frame rolling fatigue trend (solid red curve) to highlight mechanical drift over time.
* **Dynamic Threshold Integration:** Displays an active orange dashed threshold line set at $92\%$ of the peak loss value, instantly flagging points where execution deviates from the baseline model.
* **Chart Export Capability:** Features an interactive **Download Loss Chart (PNG)** button allowing researchers to export high-resolution telemetry visuals for reporting.

<img width="1477" height="755" alt="Screenshot 2026-08-30 231826" src="https://github.com/user-attachments/assets/d5d5b13e-0227-4943-a80a-bb4be67e2618" />

### 2. Detected Fatigue Spikes Table
* **Precision Frame Logging:** Programmatically captures and lists exact video frame indices, precise timestamps in seconds, and corresponding high-loss MSE values.
* **Anomaly Isolation:** Highlights critical moments of form breakdown where reconstruction loss crosses safety margins, enabling targeted frame-by-frame review.

<img width="1501" height="427" alt="Screenshot 2026-08-30 231846" src="https://github.com/user-attachments/assets/e9a2168e-d14f-4430-a78b-4ac6646cbd66" />

### 3. Automated Stride Segmentation & Breakdown
* **Cycle Extraction:** Successfully segments individual movement cycles from joint kinematics, displaying high-level metric cards for **Total Strides Detected** (`16`) and **Average Stride Duration** (`1.34 seconds`).
* **Normalized Knee Angle Profiles:** Overlays multi-colored sinusoidal curves tracking right knee flexion angles across frames within each stride cycle, revealing consistency, structural variation, and recovery phases across the session.

<img width="1586" height="891" alt="Screenshot 2026-08-30 231906" src="https://github.com/user-attachments/assets/ec75c9d8-1e4e-4480-a96d-132cbf1210de" />

### 1. Phase 2: Predictive Lead-Time Analysis
* **Early-Warning Metrics:** Evaluates predictive capability by contrasting the **Model Fatigue Warning** timestamp against the **Actual Deceleration Marker**.
* **Lead-Time Delta Calculation:** Displays the exact buffer duration (e.g., model warning at `5.1s` versus physical deceleration at `18.97s`, yielding a **13.87s early** proactive warning).
* **Quick Coaching Summary:** Generates automated natural-language coaching feedback instructing athletes on targeted endurance adjustments based on observed joint-stability thresholds.

<img width="1582" height="339" alt="Screenshot 2026-08-30 231928" src="https://github.com/user-attachments/assets/5e516173-9c7e-419f-9142-001f97c74425" />


### 2. Advanced Analytics & Enhancements Section Overview
* **Modular Dashboard Tabs:** Groups secondary diagnostic features underneath the `🚀 Advanced Analytics & Enhancements` container, split cleanly into interactive sub-tabs: **Performance Radar**, **Auto-Calibration**, and **Session Comparison**.

* The three sections for the Advanced Analytics & Enhancements Section are shown below:
  
### 3. Multi-Axis Biomechanical Radar Profile (Performance Radar Tab)
* **Holistic Trajectory Mapping:** Visualizes a multi-axis polygon chart evaluating comprehensive skater attributes, including **Knee Stability**, **Stride Consistency**, and **Recovery Speed**.

<img width="1582" height="853" alt="Screenshot 2026-08-30 232009" src="https://github.com/user-attachments/assets/70369a8c-47a0-4643-bd2c-48ddce87e6cf" />


### 4. Baseline Auto-Calibration (Auto-Calibration Tab)
* **Automated Threshold Tuning:** Features an execution button (**Auto-Calibrate Threshold from First 5s**) that establishes baseline anomaly margins dynamically from the initial frames of the active session.

<img width="1576" height="316" alt="Screenshot 2026-08-30 232019" src="https://github.com/user-attachments/assets/40c28e71-a975-4ce1-8b21-f68fa3a56c8d" />


### 5. Side-by-Side Run Comparison (Session Comparison Tab)
* **Comparative Telemetry Evaluation:** Places **Run 1 (Current Session)** peak loss statistics directly next to **Run 2 (Baseline / Previous)** mock or historical tracking data to quantify performance drift across separate trials.

<img width="1579" height="422" alt="Screenshot 2026-08-30 232036" src="https://github.com/user-attachments/assets/db9b6df3-4b1e-4f23-b5a9-f1f3a46ef9cf" />
