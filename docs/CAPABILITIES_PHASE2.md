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
* **Status:** `[PARTIALLY FUNCTIONAL — corrected 9/16]`
* **Details:**
  * **ONNX FP32 export: genuinely functional, verified 9/16.** Fixed a
    broken export pipeline (previous attempt used PyTorch's newer
    "dynamo" exporter, which produced symbolic shape mismatches
    incompatible with LSTM models). Switched to the legacy TorchScript
    exporter (`dynamo=False`). Verified numerically equivalent to the
    PyTorch model (max output difference: 0.000031). **Measured 3.08x
    faster inference than PyTorch** (0.918ms vs 2.824ms mean, batch
    size 8, 200 runs, CPU) — see `benchmark_onnx_speed.py` for the
    reproducible benchmark. `onnx_inference.py` provides a real
    `ONNXFatigueDetector` class that actually calls
    `onnxruntime.InferenceSession` — previously `onnxruntime` was
    imported but never invoked anywhere in the codebase.
  * **ONNX INT8 quantization: file-size bug fixed, but introduces a
    correctness-breaking numerical bug.** The previous "int8" file was
    larger than the FP32 original (626KB vs 133KB) due to a broken
    quantization process. Now genuinely smaller (144KB vs 495KB, a real
    70.9% reduction) after switching exporters and adding proper
    pre-processing. However, output verification shows a max difference
    of 55.7 vs. PyTorch — dynamic INT8 quantization of this model's LSTM
    recurrent weight matrices produces functionally broken output, not
    a normal accuracy/speed tradeoff. **INT8 quantization is not usable
    in its current form and should not be used for inference.** This is
    a known, documented limitation of dynamic quantization on LSTM
    architectures, not a data or process error — a fix would likely
    require static (calibration-based) quantization or per-channel
    weight handling specific to recurrent layers, not yet attempted.

### ⚙️ Advanced Fatigue Detection Sensitivity
* **Status:** `[COMPLETED]`
* **Details:** Added adjustable threshold peak multipliers allowing custom fine-tuning of reconstruction loss anomaly detection.

### 📊 Temporal Window Refactoring
* **Status:** `[COMPLETED]`
* **Details:** Upgraded comparative research visualization logic, replacing global indexing with explicit absolute frame and timestamp mapping.

### 🏆 ACSEF Competition Submission & Finalization
* **Status:** `[In progress]`
