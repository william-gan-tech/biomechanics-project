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
 
## 8/17: Label Balancing & Pipeline Synchronization

- **Action Taken:**
  - **Label Distribution Optimization:** Investigated the "all positive" label issue (80/80) where the pre-deceleration detection was too sensitive. Increased the `drop_threshold` from `0.03` to `0.10` in both `train_model.py` and `evaluate_model.py` to better isolate meaningful biomechanical degradation from normal movement fluctuations.
  - **Supervised Pipeline Debugging:** Executed multiple training/evaluation iterations to synchronize the binary classifier's performance. Analyzed confusion matrices and classification reports to diagnose model convergence patterns, identifying that the dataset size for Skater A is currently the primary constraint for label variability.
  - **Journaling & Planning:** Formulated a structured plan to resolve the visualization bug identified on 8/16. Logged current technical roadblocks in the development journal for tomorrow's focused debugging session.

- **Problems & Decisions:**
  1. **Binary Classification Sensitivity:** Despite increasing the `drop_threshold` to `0.10` and then `0.26`, the model output remained biased toward a single class. This confirmed that the current windowing strategy on the short `skater_a` dataset effectively classifies nearly every window as a "drop." 
  2. **Deferred Visualization Fix:** Acknowledged that the identical curves in `compare_research.py` are a result of improper window indexing (pulling global defaults instead of segmented temporal phases). This is queued as the priority task for the next development window.

- **💡 Strategic Pivot for Tomorrow:**
  - **Task 1: Resolve Visualization Logic:** Refactor `compare_research.py` to move away from global indexing. I will implement an explicit index-mapping function that separates "Fresh" (first 20% of data) and "Fatigued" (last 20%) phases to ensure the multi-panel plots accurately represent kinematic divergence.
  - **Task 2: Evaluate Labeling Logic:** Review `src/label_pre_deceleration.py`. If threshold adjustments continue to fail, I will revise the labeling logic to use a relative window-to-window comparison (percentile change) rather than an absolute threshold to better capture true "pre-deceleration" events across different movement speeds.

## 8/18: Long-Form Video Ingestion, Absolute Frame Mapping & Comparative Visualization Success

- **Action Taken:**
  - **Long-Form Video Ingestion:** Successfully transitioned from short-clip processing to a robust, long-form pipeline capable of handling 6-minute time trial videos (`skater_time_trial.mp4`).
  - **Absolute Frame Mapping:** Refactored the preprocessing logic to use hard-coded, frame-precise segment extraction (Fresh State: frames 500–1250; Fatigued State: frames 5625–6350) based on 25 FPS video synchronization, replacing faulty global-index slicing.
  - **Visualization Logic Resolution:** Successfully debugged `compare_research.py` by implementing explicit temporal segment mapping. The script now correctly isolates and displays independent curves for fresh vs. fatigued movement states, confirming clear biomechanical divergence.
  - **Project Documentation & Roadmap:** Updated `abilities.md` and finalized the milestone report (`run_research_milestone.py`) to reflect these new capabilities.

- **Problems, Challenges & Decisions:**
  1. **Visualization Convergence (Identical Plots):** The previous "identical curve" bug was caused by the script defaulting to the same index range for both datasets. I solved this by implementing an explicit, hard-coded frame-range filter in `preprocess_video.py` that saves separate CSV files for each distinct performance state, ensuring the visualization logic pulls two completely different datasets.
  2. **Pipeline Scaling for Long-Form Media:** Initial attempts to run the full 6-minute video failed due to pathing issues and improper loading of the larger video file. I resolved this by verifying the file path within the project's `data/` directory and utilizing MediaPipe's high-efficiency PoseLandmarker to process the longer sequence frame-by-frame, confirming that the model's memory footprint remained stable even with a significantly larger input file.

- **💡 Strategic Milestone Accomplishment:**
  - Today’s progress proves that my pipeline can handle high-density, long-form athletic data. The clear visual divergence in the comparative plots (knee flexion/postural displacement) provides the foundational evidence needed to argue that biomechanical degradation *can* be mathematically anticipated before visible performance collapse occurs.
 
## 8/19: Multi-Skater Expansion, Dashboard Re-branding & Pipeline Verification
- **Action Taken:**
  - **Pipeline Expansion:** Successfully ingested and processed long-form skating footage of elite athlete Mia Manganello Kilburg.
  - **Data Standardization:** Performed systematic file renaming and directory organization to standardize multi-subject naming conventions (`mia_fresh.csv`, `mia_fatigued.csv`, `subject_meek_fresh.csv`, etc.).
  - **Dashboard Modernization:** Updated `src/dashboard.py` to support dynamic skater selection, allowing seamless real-time switching between Mia Manganello Kilburg, Patrick Meek, and reference skater Sven Kramer. 
  - **Verification:** Validated that the batch evaluation script (`batch_evaluate_skaters.py`) correctly auto-discovers all CSV files in the `data/` directory, ensuring scalability for future subjects.
  - **Environment Audit:** Verified that all core preprocessing scripts are correctly mapping frames from long-form video to individual kinematic CSVs, ensuring cross-subject consistency in the anomaly detection pipeline.

- **Problems, Challenges & Decisions:**
  1. **Subject Discovery & Dashboard Sync:** Initial difficulty in accessing Patrick Meek’s data was traced to a mismatch between current file naming in the `data/` folder and the dashboard's hardcoded dropdown logic. Decided to keep the dropdown logic flexible and ensure all future skater additions follow the `[name]_[state].csv` convention to allow the batch processor and dashboard to detect them automatically.
  2. **Pipeline Scalability:** Confirmed that the current batch processing architecture (`batch_evaluate_skaters.py`) is successfully decoupled from specific skater names, meaning adding new subjects requires zero core code changes—only file ingestion and directory population.
  3. **Data Integrity:** Ensured that Mia’s data, now representing a new elite baseline, is correctly separated from Meek's, providing a robust dataset for comparing kinematic divergence between different world-class performance profiles.

- **💡 Strategic Milestone Accomplishment:**
  - The project has successfully moved from a single-skater research tool to a multi-subject comparative platform. By integrating Mia Manganello Kilburg’s data, I now have a diverse elite baseline, which significantly strengthens the research's ability to generalize biomechanical degradation models across different athletic profiles. The dashboard is now fully functional and ready for multi-skater comparative auditing.


## 8/20: YouTube Integration Attempts, Haralds Silovs Addition & External Dependency Friction
- **Action Taken:** 
  - Expanded the Streamlit web application's "Form & Technique" baseline profile mode to incorporate Haralds Silovs alongside existing reference athletes Sven Kramer and Jorrit Bergsma.
  - Developed a dedicated downloader script named `download_videos.py` utilizing the Python media extraction library `yt-dlp` to automate the local acquisition of external biomechanical reference videos.
  - Encountered major runtime roadblocks during script execution due to streaming security updates, requiring external JavaScript runtimes like Deno for format extraction.
  - Pivoted away from local video scraping, discarding the brittle `yt-dlp` dependency in favor of using official YouTube embed players via Streamlit's native `st.video()` component to ensure stable cloud and local playback.
  - Reverted the dashboard code to its structured 400+ line baseline temporarily to clear out scraper artifacts while mapping out direct web-link integration logic.

- **Problems, Challenges & Decisions:**
  1. **Format Extraction & Runtime Errors:** Running `download_videos.py` triggered terminal warnings stating that no supported JavaScript runtime could be found, accompanied by a strict deprecation advisory for YouTube format extraction without Deno or Node.js. This caused a cascading extractor error reporting that the requested video formats were unavailable.
  2. **Avoiding Brittle Dependencies:** Relying on automated local video scraping libraries for science fair demonstrations introduces unnecessary friction and high risk of unexpected breakage when platform backend protocols change. Debugging would require configuring external environment variables and local runtimes, which compromises project reliability.
  3. **Strategic Architectural Pivot:** Decided to eliminate local file downloading entirely. By leveraging direct video streaming URLs via native Streamlit embed components, the dashboard bypasses file-system overhead and format parsing errors entirely, guaranteeing an evergreen and robust user interface.

- **💡 Strategic Plan for Future:**
  - **Task 1: Native Embed Deployment:** Embed official YouTube video URLs directly into the `dashboard.py` interface for all three reference skaters to guarantee smooth, error-free media playback.
  - **Task 2: Dynamic UI Refinement:** Link the embedded media player and analytical metric cards cleanly to the sidebar selection state so that switching between Sven Kramer, Jorrit Bergsma, and Haralds Silovs dynamically updates both the visual video feed and corresponding charts.
  - **Task 3: Full System Verification:** Run a comprehensive local test suite of the Streamlit application to ensure zero lingering path errors or broken dependencies remain from the media downloader experiment.


## 8/22: Phase 1 Finalization, Roadmap Structuring & Documentation Polish
- **Action Taken:**
  - **Phase 1 Finalization & Summary:** Formally locked in the completion of Phase 1, solidifying the proof-of-concept that deep learning autoencoders and LSTM architectures can successfully utilize comparative temporal joint-angle trajectories across segmented clips to proactively forecast biomechanical performance degradation.
  - **Phase 2 Roadmap Definition:** Outlined and structured the core objectives for Phase 2, shifting focus from manual segmenting to automated end-to-end video ingestion and stride extraction.
  - **Documentation & Journal Sync:** Updated global project references, capability logs (`abilities.md`), and the development journal to accurately reflect the clean distinction between Phase 1 achievements and Phase 2 development plans.

- **Problems, Challenges & Decisions:**
  - **Distinguishing Research Phases:** A key challenge in documenting a multi-phase research project is preventing confusion between completed proof-of-concept components and future automation goals. Cleanly separated the project description into distinct operational tiers so that external reviewers and judges can immediately understand what has been experimentally verified versus what is currently being engineered.

- **💡 Strategic Direction Moving Forward:**
  - **Focus on Automation:** With Phase 1 successfully answering the primary research question using segmented data, all engineering efforts will now pivot entirely toward Phase 2's automated continuous video pipeline and stride-detection algorithms.
 
## 8/23: Automated Video Ingestion Engine, Streamlit UI Integration & Real-Time Fatigue Detection

- **Action Taken:**
  - **Automated Ingestion Pipeline (`pipeline_engine.py`):** Successfully built and integrated an end-to-end automated processing engine capable of taking raw, unsegmented MP4 video files, extracting frame-by-frame MediaPipe pose keypoints, computing multi-joint angles, and executing real-time autoencoder reconstruction loss calculations.
  - **Streamlit Dashboard Web App (`dashboard.py`):** Transitioned the project's front end from static multi-subject CSV viewing to an interactive web application featuring an **"Auto-Digest New Video (Upload)"** mode. Users can now upload raw skating trials directly in the browser and instantly trigger full pipeline execution.
  - **Dynamic Statistical Thresholding ($Mean + 2.0 \times Std$):** Automated the detection of form breakdown by programmatically calculating sports-science anomaly thresholds from the initial baseline movement frames, replacing manual slider guesswork with a dynamic 95% confidence interval boundary.
  - **Real-Time Telemetry & Anomaly Visualization:** Configured the dashboard to automatically render rolling reconstruction error timeline charts with dynamic threshold lines, summary metric cards, and structured data tables outlining exact timestamps of predicted fatigue spikes.

- **Problems, Challenges & Decisions:**
  1. **Resolving Module Path Resolution Errors:** Encountered a `ModuleNotFoundError: No module named 'utils'` and `ModuleNotFoundError: No module named 'src'` when launching Streamlit from nested workspace subdirectories. Resolved this permanently by standardizing local execution commands to target the root directory explicitly (`python -m streamlit run src/dashboard.py`), ensuring Python's relative import paths correctly locate the backend utility packages.
  2. **Bridging Backend Engines with Frontend UI:** Raw processing scripts previously output disconnected CSV files and static matplotlib images. Successfully refactored the pipeline engine to return serialized dataframes and dynamic figures directly to the Streamlit session state, providing a seamless, real-time user experience without requiring manual intermediate file handling.

- **💡 Strategic Milestone Accomplishment:**
  - Today’s progress marks the official transition from offline research scripts to a fully realized, automated product ecosystem. By successfully merging the computer vision pose-extraction pipeline with the live Streamlit dashboard, the project has evolved into a practical tool capable of ingesting raw footage and delivering instant, objective fatigue diagnostics.
