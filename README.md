As a skater, I’ve been to many competitions and watched top athletes race. One of the things I always did as a kid was watch a skater’s form and expression to guess if they would slow down and get passed by others. I personally found patterns that showed someone would slow down, whether it was their form, panting, or just the feeling of it. Now I’m wondering if an AI model can do the same after learning about robots and AI after being on a team and competing at VEX World Championships.

Research Question: To what extent can deep learning models leverage temporal joint-angle trajectories to anticipate biomechanical performance degradation prior to measurable athletic deceleration in speed skaters?

Drawing from competitive experience in speed skating and robotics (VEX), this project investigates whether subtle breakdowns in athletic form can be predicted using unsupervised machine learning. Rather than waiting for a skater to visibly slow down, this system uses an Autoencoder neural network to detect early micro-deviations in knee-joint kinematics, acting as an early-warning anomaly detector for physical fatigue.

  Methodology:

    The analytical pipeline transforms raw video footage into quantitative fatigue metrics through four distinct processing stages:

    Computer Vision & Pose Estimation (MediaPipe): - Extracts 3D spatial body landmarks from raw MP4 video inputs frame-by-frame.

    Isolates lower-body coordinate landmarks (hips, knees, and ankles) to compute exact joint angles dynamically.

    Signal Noise Reduction (Butterworth Filter): - Raw computer vision tracking can introduce high-frequency jitter or frame-to-frame noise.

    A digital low-pass Butterworth filter smooths the kinematic trajectories to generate clean, continuous biomechanical curves.

    Temporal Segmentation (Sliding Windows): - The continuous time-series angle data is sliced into overlapping 30-frame temporal chunks (representing individual stride cycles) with a step size of 5 frames.

    Deep Autoencoder Reconstruction (PyTorch): - An unsupervised neural network is trained exclusively on the baseline form captured at the start of the video (when the skater is fresh).

    The model compresses the 30-frame window into a bottleneck latent space and attempts to reconstruct it.

    Anomaly Scoring: By calculating the Mean Squared Error (MSE) between input strides and network reconstructions, the system outputs a quantitative fatigue score. Higher error values indicate that the kinematic pattern        has drifted away from the fresh baseline form. 

Results & Fatigue Trend Analysis
The model analyzes stride windows sequentially across the video timeline, mapping out form degradation over time.

    Baseline Phase (Start of Video): Low reconstruction error (~0.32–0.45 MSE) as the model easily recognizes clean, consistent stride patterns.

    Fatigue Phase (Later Windows): Significant upward drift in reconstruction error, peaking around window index 52 with an anomaly score exceeding 0.67, highlighting a structural breakdown in stride mechanics.

Future Work & Scalability Roadmap
To transition this research from an offline analytical model into an active, real-time coaching ecosystem, future development will focus on four primary engineering and machine learning milestones:

1. Interactive User Interface & Web Deployment
Goal: Develop a user-facing dashboard using lightweight frameworks like Streamlit or Gradio.

Impact: Allows coaches, athletes, and physical therapists without programming backgrounds to upload session videos or connect live camera feeds to visualize dynamic fatigue curves instantly.

2. Real-Time Inference & Edge Alerting
Goal: Optimize the Python processing pipeline to handle live frame-by-frame inference via OpenCV.

Impact: Moves the system from post-practice analysis to active injury prevention, triggering immediate visual or audio alerts the moment a skater's reconstruction error crosses the critical anomaly threshold.

3. Joint-Specific Error Attribution & Explainability
Goal: Deconstruct the aggregate autoencoder reconstruction error into isolated, per-joint contributions (e.g., separating knee flexion error from hip or arm swing deviations).

Impact: Provides diagnostic clarity, allowing the system to isolate the exact mechanical failure driving the anomaly score rather than outputting a generic warning.

4. Elite-Benchmark Form Guidance & Augmented Visual Feedback
Goal: Establish a "gold-standard" kinematic template derived from professional skaters, coupled with a rule-based coaching feedback engine.

Impact: Translates raw reconstruction errors into actionable corrective cues (e.g., "Bend knees deeper"), rendered directly onto video frames using color-coded OpenCV skeleton overlays to guide real-time form correction.

5. Cross-Subject Generalization & Validation
Goal: Validate the autoencoder threshold logic across a diverse cohort of athletes with varying heights, body proportions, and skating styles.

Impact: Ensures the model remains robust and generalizable across different subjects, preventing overfitting to a single individual's baseline mechanics.
