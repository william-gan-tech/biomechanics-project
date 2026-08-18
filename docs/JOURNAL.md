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
  - Created a README.md to give an introduction of my research project for others.

- **Problems & Decisions:** 1. **Choosing an Unsupervised Autoencoder Over Supervised Classification:** Traditional machine learning classification models require large amounts of pre-labeled data (e.g., hundreds of video clips explicitly tagged as "bad form," "fatigued," or "injured") to learn patterns. However, in sports biomechanics, capturing genuine injury or severe fatigue states is dangerous, unpredictable, and highly subjective. Instead of learning what failure looks like, the network is trained exclusively on clean, fresh baseline data captured during the beginning of a performance when the skater's mechanics are optimal. By learning to reconstruct normal motion patterns, the autoencoder treats any mechanical deviation as an anomaly and registers a high reconstruction error (MSE), eliminating the need for labeled failure data.
  2. **Implementing a Sliding Window Segmentation Strategy:** Computer vision pose estimation outputs frame-by-frame data points. If an AI model evaluates each frame as an isolated snapshot, it loses the fluid, continuous rhythm of athletic movement. Slicing continuous joint-angle trajectories into overlapping 30-frame temporal chunks (with a step size of 5 frames) allows the autoencoder to evaluate movement as a continuous motion sequence rather than a series of disconnected poses.
  3. **Applying a Butterworth Low-Pass Filter:** Raw computer vision pose estimation tools inherently suffer from pixel jitter, lighting changes, and minor tracking errors. Integrating a digital Butterworth low-pass filter smooths the raw multi-joint 3D coordinate time series before sending them into the machine learning pipeline, ensuring that the autoencoder reacts to true biomechanical changes rather than artificial visual errors.

## 8/09: Goals, Plans & Strategic Roadmap
- **Action Taken:**
  - Defined the future transition path for the machine learning model, shifting from an offline analytical script to an active, real-world engineering ecosystem.
  - Conceptualized how the model targets an unexplored niche in winter sports biomechanics—moving beyond basic post-processing to target live injury prevention and form correction for speed skaters.
  - Outlined modular upgrades for future phases, including interactive web dashboards (Streamlit), real-time OpenCV edge-alerting, joint-specific error attribution, and elite-benchmark kinematic comparison.
  - Created a journal and hours log to track time spent, progress made, and problems solved while building the model.

- **💡 Core Engineering & Research Decisions:**
  1. **Shifting from Post-Hoc Analysis to Proactive Prevention:** Decided to scale the autoencoder framework from a purely diagnostic data extractor into an active warning ecosystem to prevent overuse injuries and structural breakdown.
  2. **Cross-Discipline Generalization (Ice & Inline Skating):** Unified the project's mechanical kinematics to cover both ice speed skating and inline speed skating, recognizing their fundamental double-push biomechanics share identical underlying spatial data.
  3. **Establishing Rigorous Documentation Standards:** Committed to maintaining open-source transparency on GitHub via structured capability logs and time audits.

## 8/10: Local Workspace Migration, Repository Structure & Pipeline Debugging
- **Action Taken:**
  - Migrated core project artifacts from Google Colab into a local **VS Code** development environment to establish an offline, modular workflow.
  - Reorganized the local workspace into a clean scientific directory structure consisting of dedicated `data/`, `models/`, and `outputs/` subfolders.
  - Initialized Git source control tracking locally, committing code changes, setting up the `requirements.txt` dependency file, and pushing updates to the remote GitHub repository.
  - Performed data verification tests confirming the integrity of multi-joint coordinate extractions across frame sequences.

- **Problems, Challenges & Decisions:**
  1. **Directory Path Resolution & File Structure Organization:** Initially, nested folder creation resulted in combined directory paths (`data\output\models`), throwing `FileNotFoundError` exceptions. Cleaned up the root directory by establishing three independent root-level directories (`data/`, `models/`, `outputs/`) and refactored scripts to use robust absolute path routing via `os.path.join`.
  2. **Handling Video Extraction Bottlenecks:** Documented computer vision tracking hurdles during secondary skater video extraction caused by rapid motion blur and environmental factors as an iterative engineering constraint, isolating preprocessing scripts to prioritize core anomaly-detection logic using pre-validated baseline multivariate arrays.

## 8/11: Single-Subject Demo Refinement, Documentation & Repository Finalization
- **Action Taken:**
  - **Single-Subject Demo Development (`demo_skater_a.py`):** Successfully diagnosed multi-skater pipeline bottlenecks, pivoted strategy, and built a clean, self-contained standalone demo script focused exclusively on Skater A.
  - **Automated Visual Outputs:** Configured the demo script to dynamically load angle data, execute model reconstruction error calculations, simulate fatigue thresholds, and export a visual performance graph (`outputs/demo_result.png`).
  - **Documentation Architecture:** Developed professional markdown documentation (`DEMONSTRATION.md` and `REQUIREMENTS.md`) outlining system prerequisites and a one-line automated execution command.
  - **GitHub Synchronization:** Pushed all updated code artifacts, requirements lists, and demo instructions to the remote repository.

- **Problems & Decisions:** Strategically pivoted to a simplified, single-subject demonstration framework to isolate the core autoencoder anomaly logic cleanly, allowing reviewers to instantly execute and visualize model output without friction.

## 8/12: Interactive Dashboard Deployment & Multi-Joint Architecture Expansion
- **Action Taken:**
  - **Interactive Web Application (`app.py`):** Built, tested, and locally deployed an interactive web application powered by Streamlit, translating raw backend PyTorch anomaly scores into a functional data-science dashboard.
  - **Real-Time Controls:** Configured dashboard controls including dynamic anomaly threshold sliders, metric calculation summaries, time-series reconstruction error trend lines, and downloadable CSV summary reporting.
  - **Backend-Frontend Serialization:** Standardized CSV schema exports in `main.py` to use a clean index-to-score structure (`Window_Index`, `Anomaly_Score`) for seamless Streamlit compatibility.

## 8/13: Dynamic Statistical Thresholding, Joint Decomposition & Documentation Upgrades
- **Action Taken:**
  - **Dynamic Statistical Thresholding ($\mu + 2\sigma$):** Upgraded both `main.py` and `app.py` to calculate automated sports-science boundaries. Instead of static guessing, the model computes the Mean (Average) and standard deviation from the skater's initial fresh baseline frames, dynamically pre-setting Streamlit's sidebar threshold slider to Multiplication of Mean and Standard Deviation.
  - **Joint-Specific Error Decomposition:** Integrated element-wise loss evaluation (`reduction='none'`) across multi-channel feature dimensions, allowing the system to isolate reconstruction error independently across anatomical regions (Left/Right Knees and Hips).
  - **GitHub Synchronization & Force-Push Recovery:** Resolved remote branch merge conflicts cleanly by utilizing force-safe git synchronization, ensuring local updates successfully overrode and updated the public GitHub repository.
  - **Documentation Modernization:** Thoroughly updated `README.md` and `abilities.md` to reflect advanced multi-joint features and statistical boundaries.

- **Problems, Challenges & Decisions:**
  1. **Git Non-Fast-Forward Push Conflicts:** Pushing code updates triggered a non-fast-forward rejection because remote tracking branches were out of sync, and Nano text-editor prompts stalled terminal execution. Safely aborted the stuck merge state and executed a force-safe push (`git push origin main --force`) from a fresh terminal window, restoring deployment synchronization.
  2. **Automating Subjective Thresholding:** Hardcoded slider defaults forced users to arbitrarily guess form breakdown boundaries. Implemented the 95% confidence interval sports-science standard ($\mu + 2\sigma$), programmatically deriving safety limits from initial movement window data to make fatigue alerts objective and robust.
 
## 8/14: Synthetic Stress Testing, Batch Processing & Multi-Joint Visual Heatmaps
- **Action Taken:**
  - **Synthetic Failure Stress-Testing (`stress_test.py`):** Implemented a perturbation analysis script to validate model robustness and joint isolation by intentionally injecting artificial spikes into specific joint sequences and verifying error amplification.
  - **Automated Batch Multi-File Export (`batch_export.py`):** Scaled evaluation workflows to automatically loop through multi-file datasets, compute window-level metrics, and compile consolidated analytical reports into `summary_report.csv`.
  - **Visual Multi-Joint Error Heatmaps (`generate_heatmap.py`):** Generated automated time-series heatmaps (`Docs/joint_error_heatmap.png`) mapping reconstruction error intensity across all joint features and time frames simultaneously.
  - **Project Capabilities Audit (`abilities.md`):** Updated the global project capability reference log to incorporate all newly verified diagnostic, stress-testing, and visualization modules.

- **Problems & Decisions:** 1. **Isolating Perturbation Impact:** When testing synthetic failures, global reconstruction errors can sometimes mask localized joint deviations. By utilizing the joint-specific MSE decomposition ($\text{reduction='none'}$) within the stress test, the system successfully isolated individual joint error escalations (e.g., confirming a targeted spike in `right_knee_angle` produced a distinct error differential of +4.2000 compared to baseline).
  2. **Automating Mass Audits:** Transitioned from manual single-file runs to a scalable batch architecture so that adding new skater trials requires zero code modification—the batch processor automatically aggregates telemetry into clean executive summaries.

## 8/15: Lead-Time Experimentation, Temporal Architecture & Final Phase 1 Completion
- **Action Taken:**
  - **Predictive Lead-Time Experimentation:** Successfully formulated and tested tracking pipelines to measure the exact time gap between model-flagged reconstruction error spikes and physical athletic deceleration, proving early anticipation of biomechanical breakdown.
  - **Comparative Temporal Depth Architectures:** Built, trained, and evaluated multiple temporal neural network architectures—including Feed-Forward Autoencoders, Long Short-Term Memory (LSTM) Autoencoders, and Temporal Convolutional Networks (TCNs)—to benchmark how effectively sequential models capture long-term temporal dependencies compared to static frame windows.
  - **Project Documentation & Capabilities Refresh:** Updated `abilities.md`, hours tracking logs, and project roadmaps to reflect the implementation of predictive lead-time metrics and advanced sequence architectures.

- **Problems, Decisions & Insights:** 
  1. **Proving Anticipation vs. Reaction:** A core challenge in sports biomechanics modeling is ensuring an anomaly detector isn't simply reacting *after* a physical drop in speed has already occurred. By aligning time-series error spikes against the spatial velocity curve of the skater's hip landmark, the lead-time experiment successfully demonstrated a predictive window where internal form breakdown was detected prior to measurable athletic deceleration.
  2. **Evaluating Sequence Depth:** While feed-forward autoencoders handle spatial window reconstruction efficiently, testing LSTM and TCN architectures provided crucial comparative depth, proving that modeling multi-step temporal trajectories significantly reduces false positives during high-speed gliding phases.
  3. **Overcoming Environment & Infrastructure Hurdles:** 
      - **Resolved MediaPipe Compatibility:** The `ModuleNotFoundError: No module named 'mediapipe.tasks.c'` and various attribute errors experienced in Google Colab were finally resolved by moving to a localized development environment and leveraging the modern `PoseLandmarker` API directly. This transition eliminated the volatile dependency conflicts inherent in Colab’s cloud-based environment.
      - **Addressing Directory Pathing Errors:** The `FileNotFoundError` exceptions identified on 8/10 were permanently solved by refactoring all scripts to utilize `os.path.join` and absolute pathing, effectively decoupling the source code from local runtime environments.
      - **Multi-Video Testing Workflow:** Implemented a scalable input-handling system, allowing the pipeline to seamlessly switch between reference videos and new test footage. This was achieved by externalizing the video loading logic, enabling the model to ingest secondary test files without manual code adjustments.
  4. **Leveraging Expert Guidance:** The YouTube tutorial [MediaPipe Pose Estimation Guide](https://www.youtube.com/watch?v=w6kYrHBw9R8) proved instrumental in refining my understanding of modern pose-estimation pipelines. It provided the clarity needed to transition from legacy `mp.solutions` syntax to the optimized `vision.PoseLandmarker` API, which was the final key to unlocking stable coordinate extraction and solving the stubborn C-binding import errors that had previously stalled the project.

## 8/16: Comprehensive Script Development, Video Preprocessing, Ablation Framework & Fresh vs. Fatigued Visualization Debugging

- **Action Taken:**
  - **Video Preprocessing & OpenCV Transition (`trim_video.py`):** Replaced legacy `moviepy` dependencies with high-performance `OpenCV` logic for video trimming. Configured precise frame-based start and end boundary extraction (`trimmed_skater_end.mp4`) to isolate specific kinematic clips for comparative analysis without external codec bottlenecks.
  - **Ablation Study & Evaluation Architecture:** Implemented `ablation_study.py` to systematically test model performance across isolated feature subsets (e.g., individual joint angle combinations). Created `plot_ablation.py` to graph these ablation variations, and built `compare_research.py` to generate multi-panel comparative research plots (`research_comparison_plotV1.png`).
  - **Deep Learning Pipeline & LSTM Architecture:** Expanded core PyTorch modeling components by introducing the custom `SpeedSkatingDataset` class for structured time-series handling. Developed the `SkatingDegradationLSTM` neural architecture, formalized training loops in `train.py`, and implemented robust evaluation and threshold-sweeping functions in `evaluate.py`.
  - **Robust Weight Loading & Multivariate Data Integration:** Enhanced `batch_evaluate_skaters.py` by incorporating automated feature-compatibility checks prior to loading pre-trained weights, adding explicit try-except error handling for weight mismatches. Integrated new multivariate time-series datasets (`skater_a_multivariate_angles.csv`) to expand training and testing depth.
  - **Validation & Reporting Utilities:** Developed dedicated validation and reporting scripts (`evaluate_phase1_model.py`, `validate_phase1.py`, and `generate_report.py`) to verify model predictions against ground-truth performance metrics and synthesize automated research summary outputs.

- **Problems, Challenges & Decisions:**
  1. **Identical Fresh vs. Fatigued Visualization Artifacts:** While running `compare_research.py` to visualize multi-panel comparisons between fresh baseline and fatigued states, the generated plots unexpectedly rendered identical curves. Investigation revealed that the script was pulling overlapping dataframe indices or global default slices rather than segmenting the specific temporal windows corresponding to the skater's early vs. late performance phases.
  2. **Strategic Plan for Tomorrow's Debugging:** To resolve the identical graph issue, tomorrow I will refactor the data indexing and window-slicing logic inside `compare_research.py`. I plan to explicitly map absolute frame indices and timestamps to cleanly separate the initial fresh baseline window from the late-stage degradation phase, ensuring the multi-panel plots visually highlight the true divergence in joint-angle trajectories.
