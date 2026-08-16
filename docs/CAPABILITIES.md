# 🚀 Project Capabilities & Future Roadmap (`abilities.md`)

## 🟢 Part 1: Current Pipeline Capabilities (Fully Implemented & Operational)

### 🎥 Data Ingestion & Pose Estimation
* **Raw Video Ingestion:** Automatically reads and processes video streams using OpenCV (`cv2`).
* **AI-Powered Pose Estimation:** Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker.
* **3D Euclidean Spatial Mapping:** Computes vector geometry and dot products across 3D coordinates ($x, y, z$) supplied by MediaPipe to extract true anatomical angles rather than flat 2D pixel approximations.

### 📐 Kinematic Processing & Filtering
* **Biomechanical Angle Calculation:** Extracts spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.
* **Signal Noise Reduction:** Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.
* **Sliding Window Kinematic Segmentation:** Chops continuous video streams into overlapping 30-frame temporal chunks to analyze movement dynamics over time rather than isolated frames.
* **Vectorized Sliding Window Slicing:** Utilizes high-performance array manipulation to chunk continuous time-series frames into overlapping temporal matrices without performance bottlenecks.

### 🧠 Deep Learning Anomaly Detection & Temporal Architecture (PyTorch)
* **Unsupervised Deep Learning Anomaly Detection:** Utilizes deep learning autoencoders that learn the mathematical baseline of a fresh skater's movement pattern entirely on their own, eliminating the need for pre-labeled failure data.
* **Comparative Temporal Depth Architectures:** Built and evaluated multiple architectures including Feed-Forward Autoencoders, Long Short-Term Memory (LSTM) Autoencoders, and Temporal Convolutional Networks (TCNs) to capture complex temporal sequence dependencies.
* **Unsupervised Reconstruction-Based Scoring:** Relies on Mean Squared Error (MSE) minimization across a neural bottleneck, treating poor form as an out-of-distribution anomaly.
* **Compressed Latent Space Representation:** Compresses multi-joint time-series data into a low-dimensional bottleneck layer, forcing the neural network to learn and retain only the most critical, dominant kinematic features of efficient speed skating.
* **PyTorch Device-Agnostic Computation:** Dynamically detects and leverages available hardware accelerators (running seamlessly on CPU or CUDA-enabled GPUs) via automatic device routing (`torch.device`).
* **Min-Max Feature Scaling & Normalization:** Automatically bounds raw angular movement data between 0 and 1, ensuring stable gradient descent and preventing exploding gradients during training epochs.

### 🔍 Advanced Diagnostic, Predictive & Verification Features
* **Predictive Lead-Time Experimentation:** Successfully formulated and tested lead-time tracking pipelines to measure the exact time gap between model-flagged reconstruction error spikes and physical athletic deceleration.
* **Joint-Specific Reconstruction Error Decomposition:** Upgraded the loss function (`reduction='none'`) to isolate reconstruction error per anatomical region (tracking lower-body knee flexion independently from upper-body posture) to pinpoint precise failure points.
* **Dynamic Statistical Thresholding ($\mu + 2\sigma$ or $\mu + 3\sigma$):** Automated sports-science boundary calculations that compute real-time statistical limits derived strictly from the skater's initial fresh baseline frames.
* **Synthetic Failure Stress-Testing (Perturbation Analysis):** Validated model robustness and joint isolation by intentionally injecting artificial spikes into specific joint sequences and verifying error amplification (`stress_test.py`).
* **Automated Batch Multi-File Export:** Scaled evaluation workflows to automatically loop through multi-file datasets, compute window-level metrics, and compile consolidated reports (`summary_report.csv`).
* **Visual Multi-Joint Error Heatmaps:** Generated automated time-series heatmaps (`joint_error_heatmap.png`) mapping reconstruction error intensity across all joint features and time frames simultaneously.

### 🌐 Interactive Web Dashboard Capabilities (`app.py`)
* **Live Web App Hosting:** Powered by Streamlit to serve an interactive graphical user interface directly in any web browser without requiring local Python execution from users.
* **Dynamic Threshold Adjustments:** Features an interactive sidebar slider pre-set to the automated statistical baseline, allowing coaches or researchers to modify anomaly thresholds on-the-fly.
* **Multi-Metric Executive Summaries:** Automatically computes and displays high-level analytics cards for total windows analyzed, peak anomaly scores, and baseline starting errors.
* **Interactive Time-Series Charting:** Renders responsive line charts tracking stride windows against reconstruction error trends over the duration of the video.
* **Inspectable Raw Data Tables:** Provides collapsible expansion panels containing raw Pandas dataframes for deep-dive quantitative auditing.
* **One-Click CSV Report Exports:** Includes native download buttons enabling users to export customized fatigue analysis reports directly to their local machine.

### 🛠️ Software Engineering & Data Architecture
* **Modular Codebase Design:** Separates concerns cleanly between core machine learning execution, mathematical utility scripts, and visualization layers.
* **Structured Directory Hierarchy:** Automatically organizes runtime assets into dedicated, predictable directories (`/data`, `/models`, `/outputs`, `/Docs`, `/src`).
* **Robust Error Handling & Path Safety:** Utilizes `os.path` libraries to ensure absolute path compatibility across Windows, macOS, and Linux operating systems.
* **Version-Controlled Traceability:** Fully integrated with Git and GitHub for seamless branch management, commit histories, and code tracking.
* **Comprehensive Project Tracking Logs:** Maintained active development logs including `journal.md`, `hours.md`, and this `capabilities.md` reference guide.
* **Environment Reproducibility:** Bound to an explicit dependency tracker ensuring version alignment across PyTorch, Pandas, NumPy, OpenCV, and Streamlit.

---

## 🟡 Part 2: Future Development Roadmap (Planned & In Progress)

* **⛸️ Automated Stride Segmentation:** Implementing peak-detection algorithms on hip/ankle coordinates to automatically segment video streams into individual stride cycles for stride-by-stride degradation tracking.
* **👥 User-Independent Cross-Validation:** Training models on multi-skater datasets and validating generalization performance on completely unseen athletes to ensure robust, non-overfitted anomaly detection.
* **🔬 Sensitivity Analysis & Ablation Studies:** Performing systematic joint-removal experiments to mathematically identify which specific anatomical trajectory carries the highest predictive weight for fatigue anticipation.
* **🏆 ACSEF Competition Submission & Finalization:** Compiling all technical documentation, lead-time graphs, and experimental results into a presentation-ready format for high school science fair judging.
* **📏 Automated Data Normalization:** Implementing spatial scaling modules to normalize joint coordinates across different athletes, ensuring the autoencoder evaluates pure form rather than varying body proportions or camera distances.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class, professional speed skaters through the feature extractor to create a "gold-standard" baseline folder.
* **🖥️ Real-Time Edge Deployment (UI Integration):** Transitioning from offline video processing to a live web dashboard (Streamlit/OpenCV) capable of pulling live rinkside webcam frames and issuing real-time form warnings.
* **🌐 Cross-Discipline Adaptation:** Expanding the autoencoder's training baseline to handle both ice speed skating and inline speed skating simultaneously, evaluating how well the core double-push kinematic model generalizes across different friction surfaces.
* **📊 Automated Kinetic Energy Profiling:** Integrating derivative calculations (velocity and acceleration profiles of joint angles) into the feature matrix to detect explosive power loss and fatigue-induced deceleration before visual form breakdown occurs.
* **📱 Lightweight Model Quantization:** Optimizing the PyTorch autoencoder weights for edge devices (such as NVIDIA Jetson or mobile hardware) to enable low-latency, on-device inference directly at the skating rink without heavy cloud computing dependencies.
