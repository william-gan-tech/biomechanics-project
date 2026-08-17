# 🚀 Project Capabilities & Future Roadmap (`abilities.md`)

## 🟢 Part 1: Current Pipeline Capabilities (Fully Implemented & Operational)

### 🎥 Data Ingestion & Pose Estimation
* **Raw Video Ingestion:** Processes video streams using `cv2` (OpenCV) for high-performance frame extraction.
* **Modern Pose Estimation:** Replaced legacy solutions with MediaPipe’s `PoseLandmarker` API for stable, high-fidelity 3D landmark extraction.
* **3D Euclidean Spatial Mapping:** Computes vector geometry to extract true anatomical angles.
* **Precision Video Trimming:** Implemented custom `OpenCV` trimming logic to isolate kinematic clips, eliminating external codec bottlenecks.

### 📐 Kinematic Processing & Filtering
* **Biomechanical Angle Calculation:** Extracts spatial coordinates and computes exact joint angles mathematically.
* **Signal Noise Reduction:** Applies digital Butterworth low-pass filters to eliminate pixel jitter.
* **Temporal Segmentation:** Utilizes vectorized sliding window slicing (30-frame chunks) for continuous movement analysis.

### 🧠 Deep Learning Anomaly Detection & Temporal Architecture (PyTorch)
* **Unsupervised Anomaly Detection:** Autoencoder architecture trained exclusively on "fresh" baseline data to detect biomechanical deviations via reconstruction MSE.
* **Advanced Temporal Architectures:** Built and benchmarked Feed-Forward, LSTM, and Temporal Convolutional Network (TCN) models to capture long-term kinematic dependencies.
* **Custom Dataset Handling:** Developed `SpeedSkatingDataset` class for robust multivariate time-series ingestion.
* **Min-Max Feature Scaling:** Ensures stable gradient descent and normalization across all anatomical joints.
* **Device-Agnostic Computation:** Seamlessly routes PyTorch operations between CPU and CUDA-enabled GPUs.

### 🔍 Advanced Diagnostic, Predictive & Verification Features
* **Predictive Lead-Time Experimentation:** Validated the ability to detect internal form breakdown *prior* to measurable athletic deceleration by aligning MSE spikes with hip velocity curves.
* **Joint-Specific Decomposition:** Isolates reconstruction error per anatomical region (`reduction='none'`) to pinpoint specific failure points (e.g., knee vs. hip posture).
* **Dynamic Statistical Thresholding:** Implemented automated $\mu + 2\sigma$ sports-science boundaries derived from baseline performance.
* **Synthetic Perturbation Analysis:** Validated model robustness via `stress_test.py` by injecting artificial spikes into specific joint sequences.
* **Ablation Study Framework:** Systematically evaluates feature importance by testing model performance across isolated anatomical joint subsets.
* **Automated Batch Processing:** Scaled evaluation to loop through multi-file datasets and aggregate metrics into consolidated `summary_report.csv` files.
* **Visual Diagnostics:** Generates automated time-series heatmaps and multi-panel research comparison plots (`research_comparison_plotV1.png`) for state-vs-state analysis.

### 🌐 Interactive Web Dashboard Capabilities (`app.py`)
* **Live Web App Hosting:** Streamlit-powered GUI for real-time interaction.
* **Dynamic Thresholding:** Sidebar sliders for on-the-fly sensitivity adjustment.
* **Data Auditing:** Interactive charting, raw Pandas dataframe inspection, and one-click CSV report exports.

### 🛠️ Software Engineering & Data Architecture
* **Modular Pipeline Design:** Clean separation of ML execution, mathematical utilities, and visualization modules.
* **Absolute Pathing:** Robust `os.path.join` implementation ensuring cross-platform stability.
* **Version Control:** Full Git integration with branched development and automated dependency tracking (`requirements.txt`).

---

## 🟡 Part 2: Future Development Roadmap (Planned & In Progress)

* **⛸️ Automated Stride Segmentation:** Peak-detection on hip/ankle coordinates for stride-by-stride degradation tracking.
* **👥 User-Independent Cross-Validation:** Validating generalization performance on unseen athletes.
* **🏆 ACSEF Competition Submission:** Finalizing technical documentation and presentation materials.
* **🌟 Elite Benchmark Kinematic Template Directory:** Creating "gold-standard" baselines using professional skaters.
* **🖥️ Real-Time Edge Deployment:** Transitioning from offline analysis to live rinkside webcam processing.
* **🌐 Cross-Discipline Adaptation:** Generalizing models across both ice and inline skating surfaces.
* **📊 Automated Kinetic Energy Profiling:** Integrating velocity/acceleration profiles to detect explosive power loss.
* **📱 Model Quantization:** Optimizing weights for NVIDIA Jetson/mobile deployment for on-device inference.
* **🐛 Active Debugging:** Refining `compare_research.py` indexing to ensure clear visual divergence between fresh and fatigued state trajectories in multi-panel plots.
