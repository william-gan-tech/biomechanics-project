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
