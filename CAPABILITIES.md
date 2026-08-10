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

🟡 Part 2: Future Development Roadmap (Planned & In Progress)

    📏 Automated Data Normalization: Implementing spatial scaling modules to normalize joint coordinates across different athletes, ensuring the autoencoder evaluates pure form rather than varying body proportions or camera distances.

    ⚡ Dynamic Statistical Thresholding: Moving away from static thresholds by computing a real-time statistical boundary ($\mu + 2\sigma$) derived strictly from the skater's initial fresh baseline.

    🔍 Joint-Specific Reconstruction Error Decomposition: Upgrading the loss function to isolate reconstruction error per anatomical region (e.g., tracking lower-body knee flexion independently from upper-body posture) to pinpoint precise failure points.

    ⏱️ Quantitative Lead-Time Analysis: Developing an automated script to calculate the exact number of frames and seconds your autoencoder can anticipate form breakdown prior to measurable athletic deceleration.

    🖥️ Real-Time Edge Deployment (UI Integration): Transitioning from offline video processing to a live web dashboard (Streamlit/OpenCV) capable of pulling live rinkside webcam frames and issuing real-time form warnings.
