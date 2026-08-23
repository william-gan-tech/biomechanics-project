# 🚀 Project Capabilities & Future Roadmap (`abilities.md`)

## 🟢 Part 1: Phase 1 Completed & Operational (The Proof of Concept)

### 🎥 Data Ingestion & Pose Estimation
* **Raw Video Ingestion:** Automatically reads and processes long-form video streams (such as 6-minute time trials via `skater_time_trial.mp4`) using OpenCV (`cv2`).
* **AI-Powered Pose Estimation:** Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker.
* **3D Euclidean Spatial Mapping:** Computes vector geometry and dot products across 3D coordinates ($x, y, z$) supplied by MediaPipe to extract true anatomical angles rather than flat 2D pixel approximations.

### 📐 Kinematic Processing & Segmented Filtering (Phase 1 Core)
* **Biomechanical Angle Calculation:** Extracts spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.
* **Signal Noise Reduction:** Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.
* **Vectorized Sliding Window Slicing:** Utilizes high-performance array manipulation to chunk continuous time-series frames into overlapping temporal matrices without performance bottlenecks.
* **Precision Frame-Range Preprocessing:** Configures exact frame boundary mapping (such as isolating frames 500–1250 for fresh states and frames 5625–6350 for fatigued states in long-form time trials) to extract clean comparative subsets.

### 🧠 Deep Learning Anomaly Detection & Temporal Architecture (PyTorch)
* **Unsupervised Deep Learning Anomaly Detection:** Utilizes deep learning autoencoders that learn the mathematical baseline of a fresh skater's movement pattern entirely on their own, eliminating the need for pre-labeled failure data.
* **Supervised Binary Classification (LSTM):** Implements Long Short-Term Memory (LSTM) neural networks to classify pre-deceleration movement windows and predict approaching mechanical breakdown.
* **Comparative Temporal Depth Architectures:** Built and evaluated multiple architectures including Feed-Forward Autoencoders, LSTM Autoencoders, and Temporal Convolutional Networks (TCNs) to capture complex temporal sequence dependencies.
* **Unsupervised Reconstruction-Based Scoring:** Relies on Mean Squared Error (MSE) minimization across a neural bottleneck, treating poor form as an out-of-distribution anomaly.
* **Compressed Latent Space Representation:** Compresses multi-joint time-series data into a low-dimensional bottleneck layer, forcing the neural network to learn and retain only the most critical, dominant kinematic features of efficient speed skating.
* **PyTorch Device-Agnostic Computation:** Dynamically leverages available hardware accelerators (running seamlessly on CPU or CUDA-enabled GPUs) via automatic device routing (`torch.device`).
* **Min-Max Feature Scaling & Normalization:** Automatically bounds raw angular movement data between 0 and 1, ensuring stable gradient descent and preventing exploding gradients during training epochs.

### 🔍 Advanced Diagnostic, Predictive & Verification Features
* **Predictive Lead-Time Experimentation:** Successfully formulated and tested lead-time tracking pipelines to measure the exact time gap between model-flagged reconstruction error spikes and physical athletic deceleration.
* **Joint-Specific Reconstruction Error Decomposition:** Upgraded the loss function (`reduction='none'`) to isolate reconstruction error per anatomical region (tracking lower-body knee flexion independently from upper-body posture) to pinpoint precise failure points.
* **Dynamic Statistical Thresholding ($\mu + 2\sigma$ or $\mu + 3\sigma$):** Automated sports-science boundary calculations derived strictly from initial fresh baseline frames.
* **Synthetic Failure Stress-Testing (Perturbation Analysis):** Validated model robustness and joint isolation by intentionally injecting artificial spikes into specific joint sequences and verifying error amplification (`stress_test.py`).
* **Automated Batch Multi-File Export:** Scaled evaluation workflows to loop through multi-file datasets, compute window-level metrics, and compile consolidated reports (`summary_report.csv`).
* **Visual Multi-Joint Error Heatmaps:** Generated automated time-series heatmaps (`joint_error_heatmap.png`) mapping reconstruction error intensity across all joint features and time frames simultaneously.
* **Ablation Study Framework:** Systematically tested model performance across isolated feature subsets and individual joint angle combinations to determine feature importance.

### 🌐 Interactive Web Dashboard Capabilities (`dashboard.py`)
* **Live Web App Hosting & Cloud Deployment:** Powered by Streamlit Community Cloud to serve an interactive graphical user interface directly via a public URL without requiring local Python execution.
* **Interactive Click-to-Filter Data Tables:** Fully integrated data selection features allowing users to click rows in multi-subject generalization and anatomical feature ablation tables to dynamically filter visual charts and isolate joint traces.
* **Dynamic Threshold Adjustments:** Features an interactive sidebar slider pre-set to the automated statistical baseline, allowing coaches or researchers to modify anomaly thresholds on-the-fly.
* **Multi-Metric Executive Summaries:** Automatically computes and displays high-level analytics cards for total windows analyzed, peak anomaly scores, and baseline starting errors.
* **Interactive Time-Series Charting:** Renders responsive line charts tracking stride windows against reconstruction error trends over the duration of the video.
* **Inspectable Raw Data Tables:** Provides collapsible expansion panels containing raw Pandas dataframes for deep-dive quantitative auditing.
* **One-Click CSV Report Exports:** Includes native download buttons enabling users to export customized fatigue analysis reports directly to their local machine.
* **Multi-Subject Comparative Reference Profiles:** Integrated specialized tabs enabling real-time switching between cross-skater anomaly testing, 3000m endurance comparisons, and technical baseline profiles for elite athletes (Sven Kramer, Jorrit Bergsma, Haralds Silovs).

### 🛠️ Software Engineering & Data Architecture
* **Modular Codebase Design:** Separates concerns cleanly between core machine learning execution, mathematical utility scripts, and visualization layers.
* **Structured Directory Hierarchy:** Automatically organizes runtime assets into dedicated, predictable directories (`/data`, `/models`, `/outputs`, `/src`, `/assets`).
* **Robust Error Handling & Path Safety:** Utilizes `os.path` libraries to ensure absolute path compatibility across Windows, macOS, and Linux operating systems.
* **Version-Controlled Traceability:** Fully integrated with Git, GitHub, and Git LFS for seamless branch management, tracking of large video assets, and commit histories.
* **Environment Reproducibility:** Bound to an explicit dependency tracker (`requirements.txt`) ensuring version alignment across PyTorch, Pandas, NumPy, OpenCV, and Streamlit.

### Phase 1 Milestone Achieved!
* **Milestone Summary:** Successfully processed full 6-minute speed skating time trial videos using manual frame segmenting to compare a skater's fresh state against their fatigued state. This successfully answered Phase 1's research question by proving that deep learning architectures can utilize comparative temporal joint-angle trajectories across discrete video segments to proactively forecast biomechanical performance degradation prior to observable athletic deceleration.

--- 

### 🏆 Phase 1 Completed!

* **Milestone Summary:** Successfully designed, trained, and validated the foundational deep learning pipeline by utilizing manual frame segmenting on 6-minute speed skating time trials. By isolating discrete fresh versus fatigued video segments, the autoencoder and LSTM models successfully learned optimal movement baselines and detected reconstruction error spikes. This rigorously answered Phase 1's research question, proving that deep learning architectures can effectively leverage comparative temporal joint-angle trajectories across segmented clips to proactively forecast biomechanical performance degradation before observable athletic deceleration occurs. Overall added multiple videos of elite and world class skaters for identifying form and comparing a fresh and fatigued state in 3000m skating time trials.

---

## 🟡 Part 2: Phase 2 Roadmap & In-Progress Development (Full Video Automation & Continuous Tracking)

* **🔄 End-to-End Continuous Video Pipeline (Core Phase 2 Objective):** Building an automated ingestion engine that takes an entire raw video file from start to finish, removing the need for manual segmenting and outputting a rolling fatigue timeline frame-by-frame.
* **⛸️ Automated Stride & Segment Extraction:** Developing and integrating peak-detection algorithms targeting hip, knee, and ankle coordinates to automatically parse continuous video streams into distinct stride cycles.
* **📊 Temporal Window Refactoring (`compare_research.py`):** Upgrading comparative research visualization logic to replace global indexing with explicit absolute frame and timestamp mapping, ensuring fresh vs. fatigued multi-panel curves accurately display true kinematic divergence.
* **👥 User-Independent Cross-Validation:** Training models on multi-skater datasets and validating generalization performance on completely unseen athletes to ensure robust, non-overfitted anomaly detection.
* **🔬 Advanced Sensitivity Analysis & Refined Ablation Studies:** Expanding systematic joint-removal experiments to mathematically identify which specific anatomical trajectory carries the highest predictive weight for fatigue anticipation.
* **🏆 ACSEF Competition Submission & Finalization:** Compiling all technical documentation, lead-time graphs, and experimental results into a presentation-ready format for high school science fair judging.
* **📏 Automated Data Normalization:** Implementing spatial scaling modules to normalize joint coordinates across different athletes, ensuring the autoencoder evaluates pure form rather than varying body proportions or camera distances.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class, professional speed skaters through the feature extractor to create a "gold-standard" baseline folder.
* **🖥️ Real-Time Edge Deployment (UI Integration):** Transitioning from offline video processing to a live web dashboard (Streamlit/OpenCV) capable of pulling live rinkside webcam frames and issuing real-time form warnings.
* **🌐 Cross-Discipline Adaptation:** Expanding the autoencoder's training baseline to handle both ice speed skating and inline speed skating simultaneously, evaluating how well the core double-push kinematic model generalizes across different friction surfaces.
* **📱 Lightweight Model Quantization:** Optimizing the PyTorch autoencoder weights for edge devices (such as NVIDIA Jetson or mobile hardware) to enable low-latency, on-device inference directly at the skating rink without heavy cloud computing dependencies.
