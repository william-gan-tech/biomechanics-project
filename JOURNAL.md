# Project Development Journal

## 8/06: Environment Setup
- **Action Taken:** Installed Python 3.12, Microsoft C++ Runtime, and required libraries. Created project directory, initialized `main.py`, and verified baseline execution.

## 8/07: IDE & Version Control
- **Action Taken:** Downloaded VS Code and configured the workspace for deep learning development. Created a GitHub repository (`biomechanics-project`) to track progress and version-control code updates.

## 8/08: Model Pipeline Integration, Evaluation & Completion
- **Action Taken:** - Integrated modular backend components (`data_loader.py`, `biomechanics_utils.py`) into a unified workflow.
  - Executed sliding window kinematic segmentation, generating 648 windows from multivariate joint angle arrays.
  - Trained and optimized an unsupervised Multi-Channel Autoencoder utilizing GPU acceleration (CUDA).
  - Calculated real-time reconstruction Mean Squared Error (MSE) to generate quantitative anomaly and fatigue scores.
  - Developed an automated early warning flag system to isolate the precise window index where form breakdown occurs.
  - Exported analytical data (`fatigue_results.csv`) and generated visualization artifacts (`comparative_cross-run_plot.png`).
 

Problems/Decisions: 

 1. Choosing an Unsupervised Autoencoder Over Supervised Classification:

Why an Unsupervised Autoencoder? 

Traditional machine learning classification models require large amounts of pre-labeled data (e.g., hundreds of video clips explicitly tagged as "bad form," "fatigued," or "injured") to learn patterns. However, in sports biomechanics, capturing genuine injury or severe fatigue states is dangerous, unpredictable, and highly subjective.

Instead of learning what failure looks like, the network is trained exclusively on clean, fresh baseline data captured during the beginning of a performance when the skater's mechanics are optimal. 

Why It Matters: By learning to reconstruct normal motion patterns, the autoencoder treats any mechanical deviation as an anomaly and registers a high reconstruction error (MSE). This completely eliminates the need for dangerous, impossible-to-get labeled failure data, making the model universally adaptable to any athlete.

2. Implementing a Sliding Window Segmentation Strategy

The Problem: Computer vision pose estimation outputs frame-by-frame data points. If an AI model evaluates each frame as a completely isolated snapshot, it loses the fluid, continuous rhythm of athletic movement. A skater's stride cycle depends on the momentum and relationship between past and future frames.

I implemented a multivariate sliding window segmentation strategy, slicing continuous joint-angle trajectories into overlapping 30-frame temporal chunks (with a step size of 5 frames). 

Why It Matters: This approach allows the autoencoder to evaluate movement as a continuous motion sequence rather than a series of disconnected poses. It captures temporal dependencies and rhythm shifts over time, which is essential for catching subtle pacing breakdowns.

3. Applying a Butterworth Low-Pass Filter

Raw computer vision pose estimation tools (like MediaPipe) inherently suffer from pixel jitter, lighting changes, and minor tracking errors from camera frames. This creates high-frequency noise that manifests as jagged, erratic spikes in raw joint-angle calculations. 

https://www.youtube.com/watch?v=XMXX4PP4f9Y

This video provides an in-depth look at how signal filters like the Butterworth filter refine raw motion capture data to target motion curve peaks without stripping away vital movement energy. 

Filtering out high-frequency tracking noise ensures my autoencoder is reacting to true biomechanical changes in the skater's form rather than artificial visual errors caused by camera limitations. The video taught me how I can answer my research question without being limited to raw computer vision by using the Butterworth Filter. My AI Model can reach its full potential without being blocked by high frequency tracking noises or losing the speed skater's movements and positions because of this decision. 

I integrated a digital Butterworth low-pass filter to smooth the raw multi-joint 3D coordinate time series before sending them into the machine learning pipeline. Filtering out high-frequency tracking noise ensures that my autoencoder is reacting to true biomechanical changes in the skater's form rather than artificial visual errors caused by camera limitations.
## 8/09: Goals and Plans for the future of this model, Started a journal to track progress, problems, and explain key decisions while building my model
- **Action Taken:**
- Strategic Roadmap Development: Defined the future transition path for the machine learning model, shifting from an offline analytical script to an active, real-world engineering solution.
- Real-World Impact Planning: Conceptualized how the model addresses an unexplored niche in winter sports biomechanics—moving beyond basic post-processing to target live injury prevention and form correction for speed skaters.
- System Expansion Strategy: Outlined modular upgrades for future phases, including interactive web dashboards (Streamlit), real-time OpenCV edge-alerting, joint-specific error attribution, and elite-benchmark kinematic comparison.
- Created a journal and hours log to track the time I've put into this project as well as the progress I've made and the problems I've faced (and solved) while making this model. Everything made neat and organize for the layout on Github (project and repositories).

Today's development focused on scaling the autoencoder framework beyond basic analytical data extraction, transforming the architecture into an active coaching and injury prevention ecosystem. Specifically, the system is designed to: (1) prevent injury by learning individual skater baselines and triggering real-time warnings when kinematics breach safety thresholds, and (2) provide actionable coaching insights by comparing developing athletes against elite, world-class reference forms. By addressing this unexplored niche in ice and inline speed sports biomechanics, the model successfully bridges the gap between post-practice analysis and proactive safety, demonstrating that deep temporal trajectory analysis can anticipate biomechanical performance degradation long before any measurable athletic deceleration occurs.
