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

[Raw Video (.mp4)] 
       ↓
[MediaPipe Pose Landmarker (3D Keypoints)]
       ↓
[Butterworth Low-Pass Filter (Noise Reduction)]
       ↓
[Multivariate Sliding Window Segmentation (30-frame strides)]
       ↓
[Unsupervised Autoencoder (PyTorch / GPU CUDA)]
       ↓
[Reconstruction MSE & Early Warning Flag Detection]

Results & Fatigue Trend Analysis
The model analyzes stride windows sequentially across the video timeline, mapping out form degradation over time.

    Baseline Phase (Start of Video): Low reconstruction error (~0.32–0.45 MSE) as the model easily recognizes clean, consistent stride patterns.

    Fatigue Phase (Later Windows): Significant upward drift in reconstruction error, peaking around window index 52 with an anomaly score exceeding 0.67, highlighting a structural breakdown in stride mechanics.

Why an Autoencoder? 

In competitive sports biomechanics, capturing and labeling "fatigued" or "poor" form is inherently difficult because form breakdown manifests differently for every athlete. To overcome this, the system utilizes an unsupervised Multi-Channel Autoencoder. Rather than requiring massive datasets of labeled injury or fatigue footage, the neural network is trained exclusively on clean, optimal baseline data captured at the start of the performance when the skater is fresh. By learning to reconstruct normal motion patterns, the model treats any subsequent mechanical deviation as an anomaly, making it uniquely suited for real-world fatigue detection without requiring pre-labeled failure states.

Why a Sliding Window? 

Traditional computer vision models often evaluate frames in isolation, which misses the fluid, continuous nature of athletic movement. To capture temporal dynamics, the pipeline implements a sliding window segmentation strategy. Continuous joint-angle trajectories are sliced into overlapping 30-frame temporal chunks (representing full stride cycles) with a step size of 5 frames. This temporal windowing ensures that the autoencoder evaluates movement as a continuous motion sequence rather than a series of disjointed snapshots, allowing the model to detect subtle shifts in rhythm and coordination over time.

Results & Discussion:

During empirical evaluation, the autoencoder successfully tracked form degradation across the video timeline, yielding clear quantitative distinctions between fresh and fatigued states. During the baseline phase at the start of the video, the model maintained a low, consistent reconstruction error ranging from ~0.32 to 0.45 MSE, indicating stable stride mechanics. As physical exertion increased, the model recorded a significant upward drift in reconstruction error, peaking past >0.67 MSE around window index 52. An automated warning flag system was established using a statistical threshold ($\mu + 2\sigma$), triggering an alert the moment reconstruction error exceeded baseline variance by more than two standard deviations to mark the precise point of structural form breakdown.

Future Work & Scalability:

While the current iteration of the pipeline successfully validates fatigue detection using pre-recorded video, future development will transition the system into an active, real-time coaching ecosystem. Planned architectural upgrades include migrating the pipeline to live OpenCV video streams for rinkside edge deployment, implementing joint-specific error attribution to isolate exact mechanical failures (e.g., distinguishing knee flexion error from upper-body tilt), and developing an interactive web dashboard via Streamlit. Additionally, future validation will test the model across a diverse cohort of athletes to ensure cross-subject generalizability and robust performance across varying skating styles.

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

Why This Model Matters: Post-Phase 1

Completing Phase 1 establishes a functional automated data pipeline that processes raw video into quantitative temporal kinematics and anomaly scores using an unsupervised autoencoder. This stage matters because it eliminates the subjectivity of human coaching observation, transforming video footage into objective numerical data. Instead of relying on a coach's naked eye to guess when a skater is getting tired, the system provides a reproducible, mathematical baseline of a skater's normal motion. This proves that deep learning can successfully ingest complex multi-joint movement time series and output reliable structural health metrics without requiring pre-labeled injury footage.

Why This Model Matters: Post-Phases 2 and 3

This advanced state matters because it bridges the gap between passive post-practice analysis and active, real-time safety interventions. By isolating precise anatomical failure points (such as tracking knee flexion breakdown independently from upper-body tilt) and quantifying exactly how many frames in advance the model flags fatigue before an athlete actually slows down, the system transitions into an early-warning tool. Ultimately, it demonstrates that temporal trajectory modeling can anticipate performance degradation proactively, protecting athletes from overuse injuries and providing objective, elite-level comparative insights for both ice and inline speed skating disciplines.

Impact: Ensures the model remains robust and generalizable across different subjects, preventing overfitting to a single individual's baseline mechanics.
