🚀 Project Capabilities & Future Roadmap (abilities.md)

🟢 Part 1: Current Pipeline Capabilities (Fully Implemented & Operational)

    🎥 Raw Video Ingestion: Automatically reads and processes video streams using OpenCV (cv2).

    🤖 AI-Powered Pose Estimation: Tracks 3D human body joints frame-by-frame using MediaPipe's modern PoseLandmarker.

    📐 Biomechanical Angle Calculation: Extracts 3D spatial coordinates (hips, knees, ankles) and computes exact joint angles mathematically for every frame.

    📉 Signal Noise Reduction: Passes raw joint-angle data through a digital Butterworth low-pass filter to eliminate pixel jitter and high-frequency camera noise.

    🧠 Unsupervised Deep Learning Anomaly Detection: Utilizes a Multi-Channel Autoencoder that learns the mathematical baseline of a fresh skater's movement pattern entirely on its own, eliminating the need for pre-labeled failure data.

    🎞️ Sliding Window Kinematic Segmentation: Chops continuous video streams into overlapping 30-frame temporal chunks to analyze movement dynamics over time rather than isolated frames.

    📊 Quantitative Fatigue Scoring (Reconstruction Error): Computes Mean Squared Error (MSE) between input strides and network-reconstructed strides to output an objective, measurable Anomaly Score.

    📈 Automated Data Reporting & Visual Analytics: Exports frame-by-frame anomaly metrics into a structured CSV dataset (fatigue_results.csv) and generates programmatic trend plots (fatigue_trend_plot.png).

    👥 Cross-Athlete Baseline Comparison: Compares developing athletes against elite reference models to identify structural form deviations.

    🛠️ Local VS Code & Git Integration: Successfully migrated from Google Cloud/Colab to an offline, modular local environment with a clean three-folder architecture (data/, models/, outputs/) and synchronized version control via GitHub.

    📂 Automated Pipeline Directory Structuring: Implemented robust file-path handling and dependency tracking via requirements.txt to ensure reproducible pipeline execution.

    🧠 Compressed Latent Space Representation: The autoencoder compresses multi-joint time-series data into a low-dimensional bottleneck layer, forcing the neural network to learn and retain only the most critical, dominant kinematic features of efficient speed skating.

    🔁 Temporal Sequence Modeling: Utilizing rolling window arrays that capture velocity, acceleration, and rhythm transitions between consecutive strides, allowing the model to understand the flow of movement rather than static postures.

    🧬 Advanced Algorithmic & Deep Learning Capabilities:
    
PyTorch Device-Agnostic Computation: The model dynamically detects and leverages available hardware accelerators (running seamlessly on CPU or CUDA-enabled GPUs) via automatic device routing (torch.device).

Unsupervised Reconstruction-Based Scoring: Relies on Mean Squared Error (MSE) minimization across a neural bottleneck, meaning it treats poor form as an out-of-distribution anomaly without requiring any human-labeled crash or fatigue data.

Vectorized Sliding Window Slicing: Utilizes high-performance array manipulation to chunk continuous time-series frames into overlapping temporal matrices without performance bottlenecks.

    📐 Kinematic & Mathematical Processing3D Euclidean Spatial Mapping: 
    
Computes vector geometry and dot products across 3D coordinates (x, y, z) supplied by MediaPipe to extract true anatomical angles rather than flat 2D pixel approximations.

Feature-Level Loss Preservation: Implements element-wise error tracking (reduction='none') before aggregation, enabling the pipeline to isolate error differentials across specific dimensions rather than just a global average.

    🔄 Data Engineering & Pipeline Robustness:
    
Automated Missing-Data & Fallback Handling: Built with resilient error-catching blocks that intercept missing file paths or empty input streams gracefully, preventing pipeline crashes during automated batch processing.

Min-Max Feature Scaling & Normalization: Automatically bounds raw angular movement data between 0 and 1, ensuring stable gradient descent and preventing exploding gradients during neural network training epochs.

Isolated Script Execution Modules: Decouples utility mathematics (biomechanics_utils.py), execution scripts (main.py), and visualization wrappers (app.py) so individual components can be tested independently.
🟡 Part 2: Future Development Roadmap (Planned & In Progress)

    📏 Automated Data Normalization: Implementing spatial scaling modules to normalize joint coordinates across different athletes, ensuring the autoencoder evaluates pure form rather than varying body proportions or camera distances.

    ⚡ Dynamic Statistical Thresholding: Moving away from static thresholds by computing a real-time statistical boundary ($\mu + 2\sigma$) derived strictly from the skater's initial fresh baseline.

    🔍 Joint-Specific Reconstruction Error Decomposition: Upgrading the loss function to isolate reconstruction error per anatomical region (e.g., tracking lower-body knee flexion independently from upper-body posture) to pinpoint precise failure points.

    ⏱️ Quantitative Lead-Time Analysis: Developing an automated script to calculate the exact number of frames and seconds your autoencoder can anticipate form breakdown prior to measurable athletic deceleration.

    🖥️ Real-Time Edge Deployment (UI Integration): Transitioning from offline video processing to a live web dashboard (Streamlit/OpenCV) capable of pulling live rinkside webcam frames and issuing real-time form warnings.

    🌐 Cross-Discipline Adaptation: Expanding the autoencoder's training baseline to handle both ice speed skating and inline speed skating simultaneously, evaluating how well the core double-push kinematic model generalizes across different friction surfaces.

    📊 Automated Kinetic Energy Profiling: Integrating derivative calculations (velocity and acceleration profiles of joint angles) into the feature matrix to detect explosive power loss and fatigue-induced deceleration before visual form breakdown occurs.

    📱 Lightweight Model Quantization: Optimizing the PyTorch autoencoder weights for edge devices (such as NVIDIA Jetson or mobile hardware) to enable low-latency, on-device inference directly at the skating rink without heavy cloud computing dependencies.

    🌐 Part 3: Interactive Web Dashboard Capabilities (app.py):
    
Live Web App Hosting: Powered by Streamlit to serve an interactive graphical user interface directly in any web browser without requiring local Python execution from users.

Dynamic Threshold Adjustments: Features an interactive sidebar slider allowing coaches or researchers to modify the anomaly threshold on-the-fly and instantly observe how it impacts fatigue alerts.

Multi-Metric Executive Summaries: Automatically computes and displays high-level analytics cards for total windows analyzed, peak anomaly scores, and baseline starting errors.

Interactive Time-Series Charting: Renders responsive line charts tracking stride windows against reconstruction error trends over the duration of the video.

Inspectable Raw Data Tables: Provides collapsible expansion panels containing raw Pandas dataframes for deep-dive quantitative auditing.

One-Click CSV Report Exports: Includes native download buttons enabling users to export customized fatigue analysis reports directly to their local machine.

    🛠️ Part 4: Software Engineering & Data Architecture:
    
Modular Codebase Design: Separates concerns cleanly between core machine learning execution (main.py), mathematical utility scripts (biomechanics_utils.py), and visualization layers (app.py).

Structured Directory Hierarchy: Automatically organizes runtime assets into dedicated, predictable directories (/data, /models, /outputs).

Robust Error Handling & Path Safety: Utilizes os.path libraries to ensure absolute path compatibility across Windows, macOS, and Linux operating systems.

Version-Controlled Traceability: Fully integrated with Git and GitHub for seamless branch management, commit histories, and code tracking.

Environment Reproducibility: Bound to a explicit requirements.txt dependency tracker ensuring version alignment across PyTorch, Pandas, NumPy, OpenCV, and Streamlit.
