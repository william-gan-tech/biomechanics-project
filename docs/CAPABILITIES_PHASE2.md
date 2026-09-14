# 🚀 Phase 2 Roadmap & Capabilities (`capabilities_phase2.md`)

## 🟡 Part 2: Phase 2 Roadmap & Completed/In-Progress Development 8/23 - 8/28, 2026

### 🔄 End-to-End Continuous Video Pipeline (Core Phase 2 Objective)
* **Status:** `[COMPLETED]`
* **Details:** Built an automated ingestion engine (`pipeline_engine.py`) that processes raw video files from start to finish, removing manual segmenting and outputting a rolling fatigue timeline frame-by-frame.

### 🌐 Automated YouTube Video URL Ingestion & Processing
* **Status:** `[COMPLETED]`
* **Details:** Integrated robust URL downloading support via `yt-dlp`, enabling downloading of YouTube links directly into the Streamlit auto-digest pipeline. Verified working as of 9/10 (used to download footage for 4 additional skaters).

### 🖥️ Real-Time Web Dashboard Integration & Polishing (Streamlit UI)
* **Status:** `[COMPLETED]`
* **Details:** Integrated backend auto-digestion, dynamic anomaly thresholds, persistent session state management (`st.session_state`), and CSV/PNG report downloading into `app.py`.

### 📈 Automated Baseline Calibration
* **Status:** `[COMPLETED — corrected 9/12]`
* **Details:** Calibration originally used the deprecated `mp.solutions.pose` API and silently failed (returning a hardcoded fallback value) in several real-world test cases. Rewritten 9/08 to use the MediaPipe Tasks `PoseLandmarker` API, with an added intro/title-card skip and per-frame leg-visibility fallback (9/12) after two real calibration failures were diagnosed and fixed during ablation testing.

### ⏱️ Predictive Lead-Time Analysis & Robustness
* **Status:** `[COMPLETED]`
* **Details:** Implemented lead-time delta calculations comparing model warning timestamps against actual physical deceleration markers.

### ⚡ Edge Device Optimization & ONNX Runtime Validation
* **Status:** `[ATTEMPTED — NOT integrated into live inference]`
* **Details:** `skating_model.onnx` (133KB) and `skating_model_int8.onnx` (626KB) both exist, but the "int8" file is larger than the original with mixed FP32/INT8/INT32 tensor types, indicating quantization did not cleanly replace the float weights. `onnxruntime` is imported in the dashboard but never actually invoked anywhere in the code — no ONNX inference path currently runs.

### ⚙️ Advanced Fatigue Detection Sensitivity
* **Status:** `[COMPLETED]`
* **Details:** Added adjustable threshold peak multipliers allowing custom fine-tuning of reconstruction loss anomaly detection.

### 📊 Temporal Window Refactoring
* **Status:** `[COMPLETED]`
* **Details:** Upgraded comparative research visualization logic, replacing global indexing with explicit absolute frame and timestamp mapping.

### 🏆 ACSEF Competition Submission & Finalization
* **Status:** `[In progress]`
