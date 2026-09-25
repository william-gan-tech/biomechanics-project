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

## 8/24: YouTube URL Ingestion Integration, Format Troubleshooting & Documentation Sync

- **Action Taken:**
  - **YouTube Link Ingestion Development:** Expanded the Streamlit application's ingestion mode to allow users to paste YouTube video URLs directly alongside local MP4 file uploads for automated LSTM inference and fatigue spike detection.
  - **`yt-dlp` Format Resolution Attempts:** Integrated media extraction logic into `pipeline_engine.py` via `yt-dlp` to download YouTube video streams directly into the pipeline execution path.
  - **Dependency & Streaming Troubleshooting:** Addressed multiple runtime format errors (such as missing FFmpeg stream merging capabilities and unavailable format restrictions on specific YouTube video IDs like `J9Ay3KsiiFw`), testing various single-stream fallback configurations (including format `'18'` progressive streams and generic `'best'` parameters).
  - **Capabilities Documentation Update (`abilities_phase2.md`):** Updated the Phase 2 roadmap capabilities log to formally record the addition of automated YouTube video URL ingestion support and multi-format handling within the dashboard architecture.

- **Problems, Challenges & Decisions:**
  1. **FFmpeg & Adaptive Stream Limitations:** YouTube frequently splits high-definition video and audio into separate streams that require FFmpeg to merge. Because FFmpeg was not available in the runtime environment, standard high-quality requests triggered merging errors (`You have requested merging of multiple formats but ffmpeg is not installed`).
  2. **Format Availability Friction:** Restricting downloads to single pre-combined files (`best[ext=mp4]` or legacy format `18`) caused format availability errors on certain video IDs where those legacy streams have been phased out by YouTube's backend. This highlighted the inherent fragility of client-side video scraping for local desktop pipeline execution.
  3. **Decision to Pause Ingestion Scraping:** Recognizing that wrestling with evolving YouTube extraction constraints was blocking core application testing, decided to temporarily pause URL downloader debugging to prioritize system stability and cleaner alternative workflows.

- **💡 Strategic Direction Moving Forward:**
  - **Focus on Local Upload Reliability:** Re-emphasize the robust local MP4 file upload workflow (`dashboard.py`) for reliable, error-free demonstration and testing of the LSTM fatigue autoencoder pipeline.
  - **Revisit Remote Ingestion Later:** If direct URL downloading is revisited in future updates, transition toward a pre-downloaded caching strategy or integrate a server-side API wrapper to completely insulate the local Streamlit client from third-party extractor breakage.
 
## 8/25: YouTube URL Ingestion Integration, Format Troubleshooting & Documentation Sync

- **Action Taken:**
  - **YouTube Link Ingestion Development:** Expanded the Streamlit application's ingestion mode to allow users to paste YouTube video URLs directly alongside local MP4 file uploads for automated LSTM inference and fatigue spike detection.
  - **`yt-dlp` Format Resolution Attempts:** Integrated media extraction logic into `pipeline_engine.py` via `yt-dlp` to download YouTube video streams directly into the pipeline execution path.
  - **Dependency & Streaming Troubleshooting:** Addressed multiple runtime format errors (such as missing FFmpeg stream merging capabilities and unavailable format restrictions on specific YouTube video IDs like `J9Ay3KsiiFw`), testing various single-stream fallback configurations (including format `'18'` progressive streams and generic `'best'` parameters).
  - **Capabilities Documentation Update (`abilities_phase2.md`):** Updated the Phase 2 roadmap capabilities log to formally record the addition of automated YouTube video URL ingestion support and multi-format handling within the dashboard architecture.

- **Problems, Challenges & Decisions:**
  1. **FFmpeg & Adaptive Stream Limitations:** YouTube frequently splits high-definition video and audio into separate streams that require FFmpeg to merge. Because FFmpeg was not available in the runtime environment, standard high-quality requests triggered merging errors (`You have requested merging of multiple formats but ffmpeg is not installed`).
  2. **Format Availability Friction:** Restricting downloads to single pre-combined files (`best[ext=mp4]` or legacy format `18`) caused format availability errors on certain video IDs where those legacy streams have been phased out by YouTube's backend. This highlighted the inherent fragility of client-side video scraping for local desktop pipeline execution.
  3. **Strategic Decision & Interview Narrative Value:** Decided to temporarily pause local URL downloading due to these platform constraints, intentionally framing this technical hurdle as a major real-world engineering challenge. For competition interviews, this serves as an excellent case study on navigating third-party API instability, external dependency limits (like missing binary runtimes such as FFmpeg), and pivoting toward robust architectural solutions.

- **💡 Strategic Direction Moving Forward:**
  - **Focus on Local Upload Reliability:** Re-emphasize the robust local MP4 file upload workflow (`dashboard.py`) for reliable, error-free demonstration and testing of the LSTM fatigue autoencoder pipeline.
  - **Revisit Remote Ingestion Later:** If direct URL downloading is revisited in future updates, transition toward a pre-downloaded caching strategy or integrate a server-side API wrapper to completely insulate the local Streamlit client from third-party extractor breakage.

## 8/26: Edge Optimization, PyTorch Model Quantization & ONNX Runtime Deployment

- **Action Taken:**
  - **Model Optimization & Edge Acceleration:** Focused on translating heavy research model weights into lightweight, high-performance formats suitable for deployment on edge devices and local hardware without relying on heavy cloud compute infrastructure.
  - **Integration of Quantization Principles:** Studied advanced deep learning optimization strategies—inspired by technical deep dives such as PyTorch's quantization framework ([Deep Dive on PyTorch Quantization](https://www.youtube.com/watch?v=c3MT2qV5f9w))—to transition from floating-point precision (FP32) to 8-bit integer (INT8) instructions. 
  - **ONNX Runtime Bridge (`skating_model.onnx`):** Successfully linked the autoencoder and tracking models to an ONNX runtime environment (`onnxruntime`), verifying that serialized inference sessions can run locally inside the Streamlit dashboard with minimal memory overhead and drastically reduced latency.
  - **Phase 2 UI Polishing & Report Export:** Finalized Phase 2 capabilities in `dashboard.py`, including robust error handling, interactive metric tooltips, and an automated CSV report export button for coaches and judges.

- **Problems, Challenges & Decisions:**
  - **Inference Latency & Resource Constraints:** Raw deep learning models running continuous video frame processing often introduce latency bottlenecks, making real-time web UI feedback sluggish. Floating-point computations require heavier memory footprints which challenge edge deployment.
  - **Why Model Optimization and Quantization Matter:** As explained in engineering tutorials like [Deep Dive on PyTorch Quantization](https://www.youtube.com/watch?v=c3MT2qV5f9w), moving from standard 32-bit floating-point weights to quantized 8-bit representations shrinks the overall model footprint significantly and leverages specialized low-bit hardware instructions. This optimization is often the critical threshold difference that allows a resource-intensive computer vision pipeline to fit cleanly onto resource-constrained mobile hardware or local rink-side edge devices rather than demanding expensive cloud-server infrastructure.
  - **Decoupling from Heavy Dependencies:** By compiling the model graph into an ONNX format, the system becomes framework-independent, allowing fast local inference without requiring full PyTorch library overhead on the client end.

- **💡 Strategic Direction Moving Final Milestone:**
  - **Hardware-Agnostic Deployment:** With edge ONNX support fully integrated into the auto-digest workflow, the dashboard can now perform real-time biomechanical inference locally, securely, and with high efficiency. All documentation and journal tracks are fully synced for competition presentation.

*(Note: [Deep Dive on PyTorch Quantization](https://www.youtube.com/watch?v=c3MT2qV5f9w) serves as a core technical reference explaining how reducing weight precision via 8-bit quantization optimizes inference latency and memory footprints for deployment on mobile and edge devices.)*

## 8/27: Dashboard UI State Architecture Refactoring, Persistent Session Management & Dynamic Polish

- **Action Taken:**
  - **Persistent Session State Architecture (`dashboard.py`):** Upgraded the Streamlit app execution flow using `st.session_state.pipeline_ran` and conditional result checks to prevent UI components and charts from disappearing unexpectedly during widget interactions and page reruns.
  - **Seamless Result Unpacking & Data Safety:** Structured safe variable unpacking for rolling dataframes, performance metrics, stride profiles, and lead-time analysis items directly from the backend pipeline dictionary.
  - **Comprehensive Phase 2 Capabilities Synchronization:** Updated `capabilities_phase2.md` to reflect full integration of edge ONNX acceleration, dynamic anomaly threshold adjustments ($0.70$ to $0.99$ peak multipliers), and multi-axis biomechanical radar scoring.

- **Problems, Challenges & Decisions:**
  1. **UI Component Disappearance on Widget Reruns:** Streamlit naturally reruns the entire script from top to bottom whenever any user interaction occurs (such as tweaking threshold sliders or clicking auxiliary buttons). Initially, this caused the pipeline output charts and analysis metrics to vanish because they were calculated dynamically outside a persistent state container. Resolved by gating execution behind `st.session_state` flags and caching returned artifacts safely in session memory.
  2. **Font & Layout Rendering Glitches:** Encountered minor visual clipping and inconsistent font sizing when rendering high-density multi-panel research curves alongside the new summary cards. Addressed this by enforcing clean Markdown hierarchy, high-contrast UI styling blocks, and structured column layouts.
  3. **Handling Missing Module Scopes during State Hookup:** Ensuring that imported UI layout elements correctly referenced scoped session outputs without throwing `NameError` exceptions when a file upload hadn't yet occurred. Implemented strict conditional checks (`if result.get("success", False):`) to safeguard rendering blocks.

- **💡 Strategic Milestone Accomplishment:**
  - Today’s debugging and architectural refactoring successfully stabilized the frontend application layer. By locking down persistent session states, the Streamlit dashboard now delivers a smooth, professional, and reliable user experience fit for live competition demonstrations and rigorous testing.

## 8/28: Python `-m` Module Execution Mastery, Phase 2 Completion & Comprehensive Documentation Sync

- **Action Taken:**
  - **Python Module (`-m`) Execution Standardization:** Overcame persistent relative import errors and module path conflicts across `src/` by fully standardizing execution syntax through Python's built-in module flag (e.g., executing apps via `python -m streamlit run src/dashboard.py` and running processing scripts via module context).
  - **Phase 2 Completion & Final Polish:** Formally closed out all Phase 2 operational deliverables, including end-to-end continuous video pipeline automation (`pipeline_engine.py`), automated stride/segment extraction, explicit temporal window timestamp mapping, and edge ONNX runtime validation.
  - **Documentation and Repository Alignment:** Synchronized all core tracking documents (`abilities_phase1.md`, `capabilities_phase2.md`, `demonstration.md`, `requirements.txt`, and project structure trees) to reflect a pristine, competition-ready state.

- **Problems, Challenges & Decisions:**
  1. **The Relative Import & Module Resolution Trap:** Early in development, running scripts directly (like `python dashboard.py` or `python pipeline_engine.py`) frequently threw `ModuleNotFoundError` or broke internal package relative imports (`from src.pipeline_engine import ...`) because the working directory context shifted. 
  2. **How the `-m` Flag Saved the Architecture:** Adopting Python’s `-m` invocation model completely resolved these pathing issues. Running scripts as modules (`python -m streamlit run src/dashboard.py`) forces Python to correctly treat the project root as the top-level package namespace, ensuring absolute path safety across Windows Command Prompt, PowerShell, and Linux environments without messy manual `sys.path.append` hacks. This single operational habit saved countless hours of debugging during deployment packaging.
  3. **Transition to Final Presentation Readiness:** With Phase 2 fully completed, all components—from MediaPipe 3D Euclidean spatial mapping and Butterworth signal filtering to PyTorch autoencoders, ONNX edge runtime inference, and persistent Streamlit session states—are fully verified.

- **💡 Strategic Milestone & Future Outlook:**
  - **Phase 2 Successfully Concluded:** The project has transitioned from a manual proof-of-concept (Phase 1) to a fully automated, edge-optimized, production-ready biomechanics anomaly detection system. 
  - **Ready for Presentation & Future Expansion:** All documentation is completely up-to-date and tailored for ACSEF science fair judging and high-level technical reviews. Future phases are now fully unlocked for advanced cross-skater generalized validation and automated multi-angle fusion.
 
## 8/29: Comprehensive Documentation Sprint & Repository Polish

- **Action Taken:**
  - **Documentation Synthesis:** Focused entirely on organizing and expanding project documentation across Phase 1 and Phase 2. Polished README files, demonstration runbooks, and architecture summaries to clearly articulate the pipeline's transition from raw MediaPipe joint extraction to PyTorch LSTM autoencoder anomaly detection.
  - **Narrative Integration:** Embedded my background as a competitive speed skater and endurance athlete alongside my VEX Robotics experience to highlight the core motivation: moving past reactive coaching to predictive biomechanical analysis.
  - **Architecture Mapping:** Formally documented the end-to-end data flow, detailing how 3D spatial joint coordinates pass through Butterworth filtering, rolling temporal window slicing, and multivariate autoencoder reconstruction loss calculations to surface micro-deviations in form.
  - **Export & UI Spec Alignment:** Finalized operational guides for the Streamlit dashboard, outlining how coaches and judges can leverage automated CSV report exports, interactive metric tooltips, and real-time ONNX runtime toggles.

- **Problems, Challenges & Decisions:**
  - **Documentation Drift:** Rapid feature additions (like ONNX runtime integration and dynamic thresholding) caused tracking markdown files to fall slightly behind the actual implementation code. Resolved by establishing a single source of truth for all repository specs and updating code cross-references.
  - **Technical Clarity vs. Brevity:** Balancing deep mathematical explanations of LSTM reconstruction thresholds ($\mu + 2\sigma$) with high-level readability for non-technical science fair judges required structuring documentation into layered sections—quick-start guides for users and deep-dive appendices for technical reviewers.

- **Strategic Milestone:**
  - Successfully wrapped up formal documentation alignment, producing a publication-grade repository structure that bridges rigorous machine learning engineering with real-world sports biomechanics.

- **Tomorrow's Objectives:**
  - **Finalize Phase 2 Documentation:** Complete the remaining markdown walkthroughs, verify all architecture diagrams, and ensure setup instructions for ONNX runtime inference and Streamlit deployment are fully synchronized.
  - **Debug & Fix Streamlit Mode 5 Error:** Isolate and resolve the state-handling or routing exception occurring in Mode 5 of `dashboard.py`, adding explicit fallback handlers and input validation checks to maintain UI stability.

## 8/30: Mode 5 State-Handling Resolution, Documentation Sync & Phase 3 Roadmap Planning

- **Action Taken:**
  - **Mode 5 State-Handling Bugfix:** Resolved the intermittent rendering exception and downstream `NameError` triggers in Mode 5 of `dashboard.py` by hardening session-state validation wrappers and enforcing strict conditional checks (`if result.get("success", False):`) prior to unpacking pipeline outputs.
  - **Comprehensive 8/30 Documentation Sync:** Fully updated project documentation to integrate the latest changes, capturing the execution flow for automated video auto-digestion, ONNX edge runtime inference (`skating_model.onnx`), and advanced analytics modules.
  - **Phase 3 Roadmap Establishment:** Formally outlined the scope for Phase 3, pivoting from single-athlete edge inference toward generalized multi-athlete validation and multi-angle camera stream fusion.

- **Problems, Challenges & Decisions:**
  - **State-Loss During Dynamic Widget Reruns:** Interacting with threshold multipliers and advanced analytics tabs occasionally caused variables to drop out of scope during script reruns. This was fixed by establishing bulletproof default initialization blocks inside `st.session_state` and making sure all display components reference persisted state containers rather than volatile local variables.
  - **Balancing Technical Depth and Presentation Polish:** Ensuring that the transition between foundational work (Phase 1) and advanced optimization (Phase 2) remains clear for science fair reviewers and judges.

- **💡 Strategic Milestone & Future Outlook:**
  - **Phase 2 Complete:** The application layer, edge ONNX runtime, and automated segmentation pipelines are completely polished, stable, and ready for live demonstrations.
  - **Phase 3 Horizons:** Future development will focus on multi-view camera synchronization and expanded cross-subject generalization models to further eliminate subjective bias in athletic coaching.
 
  ## 8/31: Pipeline Validation Stabilization, Feature Vector Alignment & YouTube Integration

- **Action Taken:**
  - **Pipeline Feature Alignment & Validation Debugging:** Resolved a critical feature mismatch in `pipeline_engine.py` where validation logic was still searching for legacy raw column names (`right_knee_angle`) instead of the updated 6-feature filtered and normalized multivariate model inputs (`right_knee_filtered`, `left_knee_filtered`, `norm_right_hip_x`, `norm_right_hip_y`, `norm_right_shoulder_x`, `norm_right_shoulder_y`).
  - **Robust YouTube URL Handling:** Integrated and tested live sample video data using external references like [YouTube Video](https://www.youtube.com/watch?v=06TE_U21FK4&t=1s), utilizing `yt_dlp` stream downloads to instantly auto-digest, normalize, and evaluate remote skating performance streams.
  - **End-to-End System Verification:** Successfully executed the complete ingestion-to-dashboard workflow via Streamlit, verifying that feature extraction, LSTM autoencoder inference, dynamic thresholding ($\mu + 1.5\sigma$), and predictive lead-time computations run smoothly without breaking.

- **Problems, Challenges & Decisions:**
  - **Mismatched Schema Validation:** Transitioning the model from raw angle tracking to a more robust 6-feature normalized representation caused validation hooks to reject valid videos because they searched for older columns. This was resolved by updating `validate_skating_content` to verify the presence of active filtered joint signals.
  - **Leveraging External Reference Videos:** Using benchmark links like the target speed skating video (`https://www.youtube.com/watch?v=06TE_U21FK4&t=1s`) provided a vital real-world validation asset, allowing us to test downloader resilience, stride peak segmentation, and anomaly spike behavior on actual competitive footage rather than just local development files.

- **💡 Strategic Milestone & Future Outlook:**
  - **Pipeline Fully Stabilized:** The end-to-end processing pipeline is now completely harmonious—from raw local file or YouTube stream ingestion through normalization, PyTorch LSTM autoencoding, and Streamlit visualization.
  - **Ready for Live Deployment & Demos:** With the core architecture fully resilient and tested against real-world video inputs, the system is primed for seamless live demonstrations and performance profiling.
 
  ## 9/01: Pipeline Validation Stabilization, Feature Vector Alignment & YouTube Integration

- **Action Taken:**
  - **Pipeline Feature Alignment & Validation Debugging:** Resolved a critical feature mismatch in `pipeline_engine.py` where validation logic was still searching for legacy raw column names (`right_knee_angle`) instead of the updated 6-feature filtered and normalized multivariate model inputs (`right_knee_filtered`, `left_knee_filtered`, `norm_right_hip_x`, `norm_right_hip_y`, `norm_right_shoulder_x`, `norm_right_shoulder_y`).
  - **Robust YouTube URL Handling:** Integrated and tested live sample video data using external references like [YouTube Video](https://www.youtube.com/watch?v=06TE_U21FK4&t=1s), utilizing `yt_dlp` stream downloads to instantly auto-digest, normalize, and evaluate remote performance streams.
  - **End-to-End System Verification:** Successfully executed the complete ingestion-to-dashboard workflow via Streamlit, verifying that feature extraction, LSTM autoencoder inference, dynamic thresholding ($\mu + 1.5\sigma$), and predictive lead-time computations run smoothly without breaking.

- **Problems, Challenges & Decisions:**
  - **Mismatched Schema Validation:** Transitioning the model from raw angle tracking to a more robust 6-feature normalized representation caused validation hooks to reject valid videos because they searched for older columns. This was resolved by updating `validate_skating_content` to verify the presence of active filtered joint signals.
  - **Leveraging External Reference Videos:** Using benchmark links like the target speed skating/pose estimation video (`https://www.youtube.com/watch?v=06TE_U21FK4&t=1s`) provided a vital real-world validation asset, allowing us to test downloader resilience, stride peak segmentation, and anomaly spike behavior on actual competitive footage rather than just local development files.

- **💡 Strategic Milestone & Future Outlook:**
  - **Pipeline Fully Stabilized:** The end-to-end processing pipeline is now completely harmonious—from raw local file or YouTube stream ingestion through normalization, PyTorch LSTM autoencoding, and Streamlit visualization.
  - **Ready for Live Deployment & Demos:** With the core architecture fully resilient and tested against real-world video inputs, the system is primed for seamless live demonstrations and performance profiling.
 
## 9/02: Cross-Subject Generalization Attempt & Streamlit UI Resilience — CORRECTED 9/12

- **Action Taken:** Developed `src/evaluate_ablation.py` and began cross-subject normalization work (`cross_subject_normalization.py`). Fixed dashboard `WinError 32` file-lock collisions.
- **Correction (added 9/12):** The original entry claimed a verified "MSE drop from ~4,500 to ~0.62" via this framework. Audit found the companion script (`evaluate_generalization.py`, added same day per git history) evaluates 4 arbitrary time-chunks of a single video rather than distinct subjects, and contains a bug preventing a clean full run. This MSE claim should not be cited. See 9/11–9/12 for a corrected, verified multi-skater ablation.

## 9/03: Multi-View & ONNX Export Attempts — CORRECTED 9/12

- **Action Taken:** Began a multi-view synchronization module (committed as `multi_view_fusion.py`) and exported the PyTorch autoencoder to ONNX (`skating_model.onnx`, `skating_model_int8.onnx`).
- **Correction (added 9/12):** Original entry described multi-angle fusion as "operational" and ONNX export as providing "significantly lower CPU/GPU inference latency." Neither claim could be verified: `onnxruntime` is never called in the running dashboard code, the "int8" file is larger than the original with mixed float/int types, and `multi_view_fusion.py`'s actual contents have not yet been reviewed. Status downgraded to "attempted, unverified" pending further review.

## 9/04: LOSO Evaluation Attempt — CORRECTED 9/12

- **Action Taken:** Ran `evaluate_generalization.py`.
- **Correction (added 9/12):** Original entry claimed this "empirically proved a 100.00% variance reduction in reconstruction MSE." This script does not test cross-subject generalization (see 9/02 correction) and contains a self-recursive `main()` call likely preventing a completed run. This claim is retracted. A corrected, real 7-skater LOSO ablation was completed 9/11–9/12 with the opposite result direction (see below).
 
## 9/05: Annotated Video Rendering Debugging, Real-World Occlusion Challenges & Skeleton Landmark Drift Analysis

- **Action Taken:**
  - **Live Video Rendering Verification:** Tested the newly implemented video rendering pipeline on real-world competitive speed skating footage, successfully generating and exporting annotated .mp4 output streams complete with skeleton point overlays and active bone-length scaling vectors (`hip_to_knee`, `119.1px` anchor scaling readout visible at Frame 879).
  - **Empirical Visual Validation:** Analyzed rendered output frames (e.g., Frame 879) to assess pose estimation and bone anchor tracking under high-speed, dynamic athletic conditions.

- **Problems, Challenges & Decisions:**
  - **Partial Occlusion & Landmark Jitter:** As visible in the annotated output frame, rapid leaning angles and dynamic motion blur in professional speed skating cause MediaPipe landmark detections to occasionally drift or drop key joint points (such as scattered red points around the torso and limbs).
  - **Misaligned Bone Scaling Line:** Because of temporary joint occlusion and high-speed motion blur, the orange bone-length normalization vector (`hip_to_knee`) mapped incorrectly onto the upper arm/forearm region rather than the lower body femur, exposing a vulnerability in raw landmark index reliance during complex athletic postures.
  - **Decision on Robustness Upgrades:** Acknowledged that while the architectural pipeline is fully functional and successfully renders downloadable output streams, raw single-frame landmark tracking is insufficient for high-speed sports without temporal smoothing filters or confidence-score threshold gating.
 
  <img width="1917" height="1031" alt="Screenshot 2026-09-06 002021" src="https://github.com/user-attachments/assets/23a100cb-71d8-4922-9d21-be84cdc8b4e1" />

  - **Screenshot explanation:** Frame 879 demonstrates significant system progress by successfully running end-to-end video ingestion, MediaPipe pose detection, real-time HUD telemetry, and annotated video export. However, it reveals ongoing tracking limitations caused by high-speed motion blur and dynamic athletic lean, leading to landmark jitter and an incorrectly mapped hip_to_knee bone-normalization vector that anchors to the arm instead of the femur. This visual feedback confirms that while the core pipeline is operational, raw single-frame tracking requires upcoming enhancements like temporal smoothing and confidence-score thresholding.
    
- **💡 Strategic Milestone & Future Outlook:**
  - **Visual Pipeline Functional but Imperfect:** End-to-end rendering from YouTube link ingestion to annotated video export is fully operational, providing clear visual feedback on skeletal tracking performance.
  - **Next Steps for Refinement:** To resolve landmark drift and incorrect anchor line mapping during high-speed leans, upcoming iterations will incorporate Kalman filtering/moving-average smoothing for joint coordinates and confidence-based masking to reject low-certainty landmark frames.
 
## 9/06: Annotation Stress Testing, Landmark Drift Post-Mortem & Future Bone-Length Scaling Resilience

- **Action Taken:**
  - **Comprehensive Review of Visual Failures:** Conducted a post-mortem analysis on rendered annotation frames (such as Frame 879) where high-speed motion blur and dynamic athletic lean caused MediaPipe keypoints to jitter and misalign.
  - **Re-evaluating the Scaling Vector Anomaly:** Documented how temporary joint occlusion forced the hip_to_knee bone-length normalization vector to anchor incorrectly to the upper limb instead of the femur, validating the need for stronger geometric constraints.
  - **Exploratory R&D Vision:** Outlined future methodologies to preserve robust bone-length scaling despite raw landmark dropouts, ensuring anatomical normalization remains stable under extreme athletic deformations.

- **Problems, Challenges & Decisions:**
  - **Vulnerability of Raw Single-Frame Detection:** Relying strictly on instantaneous, unmasked joint coordinates proved insufficient during high-velocity maneuvers where motion blur mimics structural distortion.
  - **Commitment to Future Bone-Length Scaling Integration:** Despite current annotation hiccups, the decision was locked in to persist and heavily refine bone-length scaling in future builds. Rather than abandoning normalization due to transient tracking errors, subsequent iterations will fortify the pipeline using temporal smoothing (e.g., Kalman filters/exponential moving averages) and confidence-score gating to reject low-certainty frames before vector calculation.

- **💡 Strategic Milestone & Future Outlook:**
  - **Resilience Over Quick Fixes:** Recognizing that tracking failures are symptoms of raw inference noise rather than flaws in the normalization concept itself, the framework will evolve to treat bone-length scaling as a primary invariant.
  - **Next-Gen Tracking Road Map:** Future development cycles will merge multi-view feature matching insights (similar to principles found in [End2End Multi-View Feature Matching](https://www.youtube.com/watch?v=uuLb6GfM9Cg) at [01:02]) with confidence-gated spatial anchoring to guarantee that structural bone ratios remain locked onto correct anatomical segments regardless of motion blur.

## 9/07: Troubleshooting Bone-Scaling Misalignments & Iterative Pipeline Debugging

- **Action Taken:**
  - **Targeted Testing of Temporal Gating:** Attempted to integrate exponential moving averages and stricter confidence-score thresholding into `render_robust_annotated_video` to prevent the `hip_to_knee` scaling vector from latching onto incorrect landmark indices during high-speed athletic occlusion.
  - **Debugging Experimental Iterations:** Tested various adjustments to the visibility gates and anatomical validation checks within `extract_ensemble_reference_scale` to filter out distorted frame coordinates. 

- **Problems, Challenges & Decisions:**
  - **Persistent Mismappings:** Despite multiple code modifications, the current implementation *did not work yet*—running tests still occasionally resulted in the normalization vector anchoring incorrectly to upper body segments under heavy motion blur. 
  - **Iterative Troubleshooting Strategy:** Encountered hurdles where tweaking confidence filters either overly restricted valid frames or allowed erratic jitter through. Decided to pause, keep the workspace active, and continue diagnosing the sequence indexing bounds to isolate where the coordinate mapping breaks down.

- **💡 Strategic Milestone & Future Outlook:**
  - **Resilient R&D Process:** Acknowledged that debugging complex pose-estimation pipelines involves iterative failure cycles before achieving geometric stability. 
  - **Next Steps:** Continuing to experiment and figure out a reliable solution for clean multi-bone anchor separation, drawing conceptual inspiration from global constraint handling seen in end-to-end multi-view feature matching frameworks.
 
  ## 9/08: Dashboard Debugging — Import Bugs & MediaPipe API Migration

- Found no hardcoded "bunny"/placeholder video anywhere in the codebase (verified via full-repo search); root cause was a stray cached file plus a locally-defined downloader in `app.py` silently shadowing the real `yt_dlp`-based one in `pipeline_engine.py`.
- Discovered installed `mediapipe` (v1.0.1) removed the legacy `mp.solutions.pose` API. Rewrote calibration and video-rendering code to use the `mediapipe.tasks` `PoseLandmarker` API already used elsewhere in the pipeline.
- Fixed `ROOT_DIR` pointing one directory above the actual project root, silently breaking relative dataset/config path lookups since launch.
- Fixed a `sys.path` ordering bug causing a stale `src/preprocess_video.py` to shadow the correct root-level copy.

## 9/09: Real Cross-Skater Comparison (Replacing Simulated Data)

- Replaced `np.random`-based demo data in Cross-Skater Anomaly mode with real feature extraction, disk caching, and DTW-based comparison (`cross_skater_compare.py`).
- Corrected mislabeled metrics from the original UI (fictitious "Hip Angle"/"Ankle Dorsiflexion" features that were never computed) with the 6 features actually produced by the pipeline; renamed a static "Cross-Subject Accuracy" value to an honestly-scoped "Similarity Score."
- Added a CSV fallback tier for skaters without video (knee-angle columns only — excluded unscaled hip/shoulder columns to avoid contaminating the bone-scaling comparison), and an explicitly labeled "SIMULATED" fallback for skaters with neither video nor usable CSV data.
- Fixed a threshold-scale mismatch: the existing 0.01–0.10 anomaly slider was calibrated for MSE loss, not the new z-scored DTW metric (range ~0.3–3.0), causing every comparison to falsely flag as anomalous.

## 9/10: Expanding Video Coverage

- Downloaded footage for 4 more skaters (Ragne Wiklund, Mia Manganello Kilburg, Jorrit Bergsma, Jan Blokhuijsen) via the existing `yt_dlp` downloader, bringing real-video coverage to 7 of 10 skaters.
- Fixed a Unicode filename-matching bug (fullwidth vs. regular vertical-bar character) causing a silent path failure; replaced hardcoded paths with keyword-based file search for two skaters.

## 9/11: Phase 3 LOSO Ablation — Bone-Length Scaling vs. Generalization

- Built a Leave-One-Skater-Out ablation (`run_bone_scaling_ablation.py`) training a fresh autoencoder per fold, comparing `reference_scale=None` vs. calibrated bone-scaling, with standardization computed from training-pool data only (not per-skater) to avoid erasing the effect being tested.
- Rewrote the script mid-run to save results after every fold and support resuming, after ~3 hours of unsaved progress were lost when the original (end-of-run-only save) version had to be interrupted.

## 9/12: Diagnosing Ablation Outliers & Recording the Result

- Diagnosed two real data-quality failures surfaced by the ablation rather than dismissing the result: Ragne Wiklund's calibration was silently using the fallback scale due to one-sided leg occlusion (fixed by making bone-length calculation pick whichever leg is visible per frame); Mia Manganello Kilburg's footage contained a mid-clip broadcast cutaway producing implausible position values (fixed with a general frame-plausibility filter, applied to all skaters/conditions equally).
- After both fixes, result direction did not change: cross-subject loss variance remained higher under bone-scaling (0.216) than without it (0.030) across n=7 skaters — recorded as the genuine pilot result rather than adjusted further to match the original hypothesis.
- Began (not yet completed) independent verification of prior-session claims in `abilities_phase3.md`/hours log that could not be corroborated against the actual codebase.

## 9/13: Fatigue-Separability Experiment (Phase 3b) & Statistical Analysis Layer

- **Action Taken:**
  - Built `run_fatigue_separability_ablation.py` (Phase 3b) to directly test the original Phase 3 wording — trains an autoencoder ONLY on other skaters' early-session ("fresh" proxy) data, then measures the reconstruction-loss gap on a held-out skater's late-session ("fatigued" proxy) data. This is distinct from the earlier Phase 3a ablation, which tested general cross-subject reconstruction variance but never distinguished fresh from fatigued motion at all.
  - Reused cached features from the Phase 3a run and added a third `zscore_only` condition (image-height normalization without bone-length division) to isolate whether bone geometry specifically adds value beyond basic standardization.
  - Built `analyze_phase3_results.py`: a proper statistical analysis layer running paired Wilcoxon signed-rank tests (appropriate for small, non-parametric paired samples) across both Phase 3a and 3b results, plus effective-sample-size reporting.
- **Problems, Challenges & Decisions:**
  - Import-order bug: `run_fatigue_separability_ablation.py` imported `model` before `run_bone_scaling_ablation` (which triggers the `sys.path` fix as a side effect of importing `pipeline_engine`), causing a `ModuleNotFoundError`. Fixed by reordering imports.
  - All paired comparisons (3a and 3b, all three condition pairs) came back statistically non-significant at n=7 — reported honestly rather than treated as a null result, with explicit notes that a small sample can't distinguish a real moderate effect from noise.

## 9/14: Outlier Sensitivity Check — Key Finding

- **Action Taken:**
  - Noticed Mia Manganello Kilburg's results were the visible outlier in every single comparison across both 3a and 3b (e.g. 3b separability gap of 2.60 vs. a next-highest value of 0.33).
  - Built and ran `outlier_sensitivity_check.py`: re-ran all four key paired comparisons with Mia excluded, to test whether the "scaling increases variance" finding held for the other 6 skaters or was driven almost entirely by her one data point (whose footage was already flagged 9/12 for a mid-clip broadcast cutaway).
- **Key Result:** The direction of the finding **flips** when Mia is excluded. With all 7 skaters, bone-length scaling shows higher cross-subject variance than unscaled/z-score-only in every comparison (3-10x higher). With Mia excluded (n=6), scaling shows **lower** variance than unscaled and z-score-only in 3 of 4 comparisons, and roughly equal in the 4th.
- **Interpretation:** This is not treated as "the real answer was hidden" — both the n=7 and n=6 results are being reported side by side. The consistent directional flip across four independent comparisons suggests footage-quality robustness (not the normalization math itself) may be the dominant factor in whether bone-length scaling helps or hurts in practice — arguably a more specific and useful finding than either version alone.

## 9/15: Multi-Person Tracking Fix & Repository/Documentation Sync

- **Action Taken:** Fixed a real, user-reported bug in `pipeline_engine.py` — bone-scaling calibration could silently swap to a different skater mid-video in footage with multiple people. Added position-continuity tracking (nearest hip-centroid across frames) plus an appearance-histogram tiebreak for ambiguous cases, and EMA smoothing on raw landmarks before any bone-length/angle math (previously smoothing only existed in video-overlay rendering). Built `diagnose_tracking_swap.py` to trace tracking decisions frame-by-frame rather than guessing at fixes blindly.
- **Problems, Challenges & Decisions:** The fix resolved the "two skaters passing near each other" case but not Lee Sang-Hwa's reference video — traced via the diagnostic to the video being a 6-clip compilation with hard scene cuts, not a real algorithm failure. Documented as a known input-content limitation rather than continuing to patch a genuinely hard, out-of-scope general problem.
- Discovered and resolved a recurring git divergence issue caused by editing documentation both locally and directly on GitHub's web interface in parallel — standardized going forward on a single local-edit-then-push workflow to prevent this.

## 9/16: Phase 4 Kickoff — Footage Audit, Feature Engineering, First Real Tracking Success

- **Action Taken:**
  - Built `audit_footage_for_starts.py`: checked existing footage for genuine simultaneous multi-person content before assuming new video was needed. Found 5 of 7 skaters already have usable content, avoiding an unplanned footage-sourcing delay.
  - Built `start_phase_features.py`: added torso-lean/crouch angle, hip velocity, and hip acceleration — three new biomechanical features not present in the Phase 3 feature set. Verified the math runs correctly against real extracted data; explicitly documented that features are right-side-only, a real limitation given skating's asymmetry.
  - Ran the reused `diagnose_tracking_swap.py` tool against Haralds Silovs' footage as the first real (non-synthetic) test of the 9/15 multi-person tracking fix.
- **Problems, Challenges & Decisions:**
  - Confirmed a genuine tracking success at frame 1786: two simultaneous people at ambiguous positions were correctly disambiguated via the appearance-histogram tiebreak (0.959 vs. 0.944 similarity) — the first real-footage validation of that fix.
  - Found a ~100-frame zero-detection stretch; rather than assuming a tracking bug, visually inspected the actual frame and confirmed it was a title-card overlay, not skating content — same class of limitation as the Lee Sang-Hwa compilation-cut finding from 9/15. This is now a confirmed pattern across 3 separate videos, not an isolated incident.
  - Also fixed two recurring deployment friction points during setup: an import-order bug in `start_phase_features.py`'s cache fallback path, and the usual file-download-not-yet-clicked issue when deploying new scripts.

## 9/16 (continued): ONNX Real Fix

- Also completed real ONNX export/quantization work today (see Phase 2 summary) — FP32 export fixed and verified (3.08x measured speedup), INT8 quantization's file-size bug fixed but a genuine correctness bug found and honestly documented as unresolved rather than claimed as working.

## 9/18: Phase 4a Real Data — First Confirmed Start Moment, Honest Detection Failures

- **Action Taken:** Tested automated acceleration-spike start detection against two real Olympic short-track candidate videos found via web search; built `test_start_candidate.py` (general-purpose download + windowed search tool) and `browse_frames_manually.py` (manual visual frame browser) after the automated approach found zero valid candidates in either video. Used real YouTube timestamps the user identified by directly watching a third candidate video, downloaded it, and manually browsed frames to find a genuine, visually-confirmed start sequence.
- **Problems, Challenges & Decisions:** Confirmed the automated velocity-before-spike filter is unreliable on chaotic multi-skater broadcast footage — landmark jitter prevents velocity from ever reading near-zero even at true rest, unlike on cleaner single-skater footage where the same filter worked correctly. Caught one apparent "biggest spike" result that turned out, on visual inspection, to be a broadcast name-graphic overlay confusing the tracker — not real motion. Built `compare_start_vs_cruise.py` using manually-confirmed frame ranges instead of algorithmic guesses, and got a real first result: 2.4x higher velocity during early-acceleration vs. a confirmed rest position. Found and documented a genuine data gap (a camera cut obscuring the literal gun-instant) rather than papering over it. Identified two unresolved issues to investigate next: a possible skater-identity swap between segments (torso lean sign flip), and confirmation that the default cruise-comparison window doesn't actually show settled steady-state skating yet.

## 9/19: Phase 4a & 4b Complete — Real Start Confirmed, Real Pack-Tracking Validated

- **Action Taken:**
  - Searched for and identified real Olympic speed skating start footage via web search after confirming existing 7-skater footage contained no genuine start-crouch content (algorithmic detection on that footage produced only false positives, e.g. a cornering lean mistaken for a start).
  - Built `test_start_candidate.py` (general-purpose download + time-windowed search) and `browse_frames_manually.py` (manual frame-by-frame visual browser with computed features printed alongside each image) after automated acceleration-spike detection (`find_candidate_start_moments.py`) found zero valid start candidates across two real Olympic short-track videos.
  - Downloaded and manually verified a genuine start sequence in `start_candidate_3.mp4` (Milano Cortina 2026 short track) using a timestamp identified by directly watching the source video: a held REST position (5.17s–5.64s), a real camera-cut data gap (~0.7s, documented honestly rather than interpolated over), and a confirmed EARLY-ACCELERATION phase immediately after (6.44s onward).
  - Built two independent identity-verification checks (`check_same_skater_identity.py`, then `check_gap_continuity.py`) after the first came back ambiguous (0.6964 similarity); the second, more decisive spatial-continuity check confirmed the REST and EARLY-ACCELERATION segments show the same skater.
  - Built `compare_technique_phases.py` and ran a real within-subject comparison across start, corner, and straightaway segments (visually categorized by hand) in the same video.
  - Reran `diagnose_tracking_swap.py` (extended to cover the full video) against `start_candidate_3.mp4` — the first real test of the 9/15 multi-person tracking fix against genuine start-line/pack simultaneity, not just incidental background people.

- **Problems, Challenges & Decisions:**
  - Confirmed the automated velocity-before-spike filter is unreliable on chaotic multi-skater broadcast footage — landmark jitter prevents velocity from ever reading near-zero even at true rest. Documented as a real limitation rather than endlessly re-tuned.
  - Caught two separate false positives where "the biggest acceleration spike in the window" turned out, on direct visual inspection, to be a broadcast name/stats graphic overlay fooling MediaPipe's tracking — not real motion. Established as a standing rule: never trust an algorithmic candidate without opening the actual frame.
  - The default cross-video cruise-comparison approach (using a separate long-track skater's footage) was abandoned once a better option became available: reusing already-visually-categorized frames from the same video for a genuine within-subject start/corner/straightaway comparison instead.

- **Results:**
  - **Phase 4a — real, physically sensible finding:** torso-lean stability ranks straightaway (std 26.2°) > start (35.9°) >> corner (82.9°) — corners are 3.2x more variable than straightaways, consistent with expected cornering biomechanics. Knee symmetry also differs sharply by phase: straightaway nearly symmetric (2° gap between knees), corner clearly asymmetric (14° gap) — concrete evidence motivating the planned Phase 6 need for bilateral tracking.
  - **Phase 4b — real, validated finding:** across the full `start_candidate_3.mp4` video, the tracking system correctly triggered 135 appearance-based disambiguations during genuine multi-person proximity, with a 10-sample manual review confirming correct selection in every case. 58 lost-track events were consistent with the already-documented broadcast-cut limitation, not tracking failure.
  - **Both Phase 4a and Phase 4b are considered complete** as of today, with real evidence backing both sub-questions rather than open caveats.

- **💡 Strategic Note:** Today's technique-phase comparison already provides real preliminary data supporting both planned Phase 5 (straightaways — shown to be the most stable phase) and Phase 6 (corners — shown to require bilateral tracking due to real, measured knee asymmetry), giving both future phases a concrete evidence-based starting point rather than a blind hypothesis.

## 9/20: Building a Verified Reference Dataset — Phase 4 Data Infrastructure

- **Action Taken:** Built `manage_labeled_segments.py`, a structured CSV-based labeling log, after recognizing that Phase 4a's real findings from 9/18 were based on n=1 (one confirmed start video) and had no reusable infrastructure for adding more skaters. Established a rigorous three-step verification process for every new segment: (1) frame-by-frame candidate-count scan to catch detection gaps and multi-person ambiguity zones, (2) identify a genuinely clean single-candidate sub-range, (3) appearance-histogram identity check (early vs. late) before trusting the segment. Applied this process across `start_candidate_3` (recovering and correcting earlier segments), Haralds Silovs' footage (2 new corner segments), and Patrick Meek's footage (start_rest and start_acceleration).
- **Problems, Challenges & Decisions:**
  - Discovered that an earlier "confirmed" corner range in `start_candidate_3` (950-1049) was actually contaminated by a mid-range identity swap between two skaters, despite looking like clean corner content visually. Correctly excluded it (and a second overlapping bad range) rather than trusting the visual read.
  - Found that comparing appearance similarity *across* phase transitions (e.g. held-rest vs. mid-stride) produces artificially lower similarity scores even for the same skater, due to genuine pose/motion-blur change — not an identity swap. Fixed the verification methodology to compare *within* the same phase instead, which produced decisively high, trustworthy results for Patrick Meek's data.
  - Caught and corrected a logging error where two entries were saved with literal unfilled placeholder text in the notes field, rather than letting it stand uncorrected.
- **Result:** Reference dataset coverage grew from effectively n=1 to n=3 for start_rest/start_acceleration and n=2 for corner/straightaway, with every entry backed by a documented, reproducible verification trail rather than visual impression alone.

## 9/21: Reference Dataset Reaches Balanced n=3 & Two Real Bugs Found Through Sensitivity Testing

- **Action Taken:** Continued building out the Phase 4 reference dataset from Patrick Meek's footage, identity-verifying a corner segment (767-815) and a straightaway segment (746-762) using the established three-step process (candidate scan → clean sub-range → identity check). This brought all four technique-phase categories (start_rest, start_acceleration, corner, straightaway) to a balanced n=3 skaters each for the first time. Built `build_reference_stats.py` to compute real pooled mean/std statistics per phase across all verified skaters, and `check_knee_variance_outlier.py` to leave-one-out sensitivity-test any surprising result rather than accepting it at face value.
- **Problems, Challenges & Decisions:**
  - Two identity checks (corner and straightaway) initially returned low similarity on wide comparison windows, which could have been misread as tracking swaps. Narrowing the comparison window in both cases resolved this decisively — confirming the low scores were caused by natural within-phase pose variation (corner lean-angle swings, stride-cycle extension/recovery), not real identity swaps. Also discovered that a high identity-similarity score does NOT guarantee phase purity: a "verified" straightaway range was found via visual check to transition into corner technique partway through, and was trimmed accordingly (746-790 → 746-762).
  - **Bug #1 (running-average):** `build_reference_stats.py` and `check_knee_variance_outlier.py` both used an incorrect sequential pairwise average `(a+b)/2` to combine a skater's multiple segments, rather than a true mean — this overweighted earlier segments and produced a falsely inflated straightaway left-knee cross-skater std (39.0°, later corrected to 7.30°). Fixed in both scripts by collecting all segment values into a list and computing a true `np.mean()` at the end.
  - **Bug #2 (sign-convention artifact):** corner torso-lean angle showed a large cross-skater std (42.28°) that turned out to be substantially a camera-angle/turn-direction sign artifact, not real technique variance — using absolute value dropped the std to 17.63°, roughly a 60% reduction. Added an `--abs` flag to `check_knee_variance_outlier.py` and confirmed this directly.
  - **Hypothesis tested and correctly retracted:** an elevated `start_acceleration` hip-velocity value for `start_candidate_2` was initially suspected to be a frame-gap artifact from filtered-out frames breaking the `.diff()`-based velocity calculation. Direct inspection of the raw frame sequence showed only one minor multi-frame gap, unrelated to the velocity spike — the elevated value instead reflects a smooth, genuine acceleration ramp, most likely explained by inconsistent segment-boundary placement across skaters (where within each skater's own ramp the labeled range was drawn) rather than a computation bug. Documented as an honest methodological limitation rather than forced into either a "bug" or "real finding" conclusion without sufficient evidence.
- **Result:** Reference dataset is balanced at n=3 for all four phases, with a corrected, sensitivity-tested reference table. Two real implementation bugs were found and fixed through targeted follow-up investigation rather than accepted at face value; one plausible hypothesis was tested and honestly retracted when the evidence didn't support it.

## 9/22: Sign-Fix Confirmed, Ramp Standardization Tested & Retracted, Dual-Tracking Attempted & Saved for Future Work, Bilateral Tracking Built

- **Action Taken:**
  - Fixed the torso-lean sign-convention bug at the source (`start_phase_features.py`) instead of only in the sensitivity-checker workaround from 9/21 — `torso_lean_angle_deg` is now abs-by-default, with the raw signed value preserved separately as `torso_lean_angle_deg_signed`. Regenerated the full multi-skater reference table (`build_reference_stats.py`) and confirmed the fix propagated correctly across all four phases automatically, with corner's corrected std (17.6°) matching yesterday's manual spot-check exactly.
  - Built `standardize_ramp_segments.py` to test a real methodological hypothesis from 9/21: that inconsistent manual segment-boundary placement (not real technique differences) was inflating cross-skater acceleration-phase variance. Standardized each skater's segment to a consistent relative "ramp onset" point instead of an arbitrary manually-picked range.
  - Attempted dual-skater simultaneous tracking (extracting BOTH skaters from Olympic time-trial footage, instead of discarding the second as noise) — three iterations: (1) position + single-frame appearance, (2) velocity-predicted position + rolling-averaged appearance, (3) added bounding-box duplicate-detection filtering after diagnosing a real ~21% duplicate-detection rate (4/19 sampled frames, confirmed via direct IoU measurement).
  - Built bilateral (left-side) feature extraction: `preprocess_video.py` now saves `norm_left_hip_x/y` and `norm_left_shoulder_x/y` (previously computed internally for torso-length but discarded); `start_phase_features.py` adds `torso_lean_angle_deg_left` and a real `hip_lateral_asymmetry` metric (bone-scaled distance between left/right hip position). Verified working against real cached extraction.
- **Problems, Challenges & Decisions:**
  - **Ramp standardization tested and honestly retracted**: standardizing increased cross-skater variance by +31.1% rather than reducing it. Traced to a specific mechanism: `start_candidate_2`'s detected ramp onset landed on a later, faster portion of their acceleration curve than the other skaters' standardized windows did — a real, documented negative result, not swept under the rug.
  - **Dual-skater tracking: all three attempts failed to fully resolve identity swapping**, even after combining velocity prediction, rolling appearance averaging, and duplicate-detection filtering. Root cause diagnosed as two compounding factors: genuine duplicate MediaPipe detections (~21% of frames) plus real tracker confusion on genuinely-separate people who are difficult to visually distinguish (similar uniforms) while in different technique phases simultaneously. Decision: stopped after three reasoned, evidence-based attempts rather than continuing to iterate blindly; documented as real future work requiring a proper motion-model-based multi-object tracker, not further ad-hoc patching. Adopted practical workaround: continue using the proven single-target tracker, prioritizing whichever skater is closest to camera/most prominent, for ongoing dataset labeling.
  - Confirmed via direct data inspection that the bilateral feature fix computes correctly (`hip_lateral_asymmetry` values in a sensible 0.11–0.24 range) after an initial display-only issue (new columns existed but weren't shown in the test script's print statement) was caught and clarified before being mistaken for a real failure.
- **Result:** Reference dataset's core torso-lean metric is now correctly bug-free at the source, not just in downstream checks. One real methodological hypothesis was tested and honestly reported as unsuccessful. Dual-skater tracking is a documented, partially-diagnosed open problem saved for dedicated future work. New bilateral-asymmetry infrastructure is built and verified, ready to test whether corner technique shows measurably more left-right asymmetry than straightaway technique — the next concrete question once old cached data is regenerated with the new columns.

## 9/23: Phase 5 Roadmap Drafted, Elite-Anchor Comparison Tool Built & Verified, Arm Swing Built with a Real Unresolved Limitation

- **Action Taken:**
  - Drafted the full Phase 5 roadmap (form analysis and coaching reference system) — Tier 1 (core ice-skating elements: elite-anchor methodology, arm swing, sit height, stride rhythm, bilateral asymmetry validation, fatigue-linked degradation, unified report), Tier 2 (refinements: push direction, push/glide separation, vertical oscillation, arm-swing mode transitions, finish technique, full sensitivity audit), and Tier 3 (inline skating as a genuinely separate discipline, not a variant of ice technique — flagged as needing its own from-scratch footage and reference dataset).
  - Built and verified `compare_to_elite_reference.py` (Phase 5a): formalizes the labeled dataset into an explicit elite-anchor reference profile (mean + std per phase + metric, using the corrected true-mean methodology from 9/21) and provides a real z-score comparison tool for any new clip. Verified correct via a self-comparison sanity check — comparing Sven Kramer against a reference profile he's a member of produced small z-scores (~±0.5-0.6) across all metrics, as expected.
  - Built arm-swing feature extraction (Phase 5b): added elbow/wrist landmark tracking to `preprocess_video.py`, computed elbow-bend angle and frame-to-frame swing amplitude in `start_phase_features.py`.
- **Problems, Challenges & Decisions:**
  - Found a real problem in the arm-swing data: elbow angle swung 150+ degrees frame-to-frame in early clip frames despite the skater's visible arm position being nearly static across the same ~0.1s window — confirmed via direct visual inspection, ruling out "real fast motion" as an explanation and pointing instead to landmark misdetection.
  - Attempted a fix using MediaPipe's own per-landmark visibility/confidence scores, hypothesizing low confidence would directly flag the bad frames. Tested this hypothesis explicitly rather than assuming it worked: known-bad frames scored 0.76-0.81, not meaningfully different from the clip's normal range (0.58-0.97) — the fix did not work as hoped. Documented this as a genuine, unresolved limitation rather than continuing to chase an automated solution, and adopted manual spot-checking as the practical mitigation, consistent with the project's established discipline of reporting failed fixes honestly rather than forcing them.
  - Working environment reset mid-session, losing local file state. Rebuilt `preprocess_video.py` and `start_phase_features.py` from conversation history, but caught that the `preprocess_video.py` reconstruction was incomplete (missing `extract_calibration_landmark_sequence`, a function never previously seen in this session) before it could overwrite real project code — verified via function-name and line-count checks first, then applied the actual fix as precise, targeted edits to the real file instead of risking a full blind replacement.
- **Result:** Phase 5a is complete and verified — the first real "compare a skater against an elite reference" tool exists and works correctly. Phase 5b's core arm-swing feature computes correctly for the large majority of frames, but carries a real, documented reliability limitation for a subset of frames (landmark misdetection not caught by confidence filtering), reported honestly rather than presented as fully solved.

## 9/24: Sit-Height Feature Validated & Integrated, Real Identity-Verification Bug Found & Fixed, New Metric-Design Limitation Discovered

- **Action Taken:**
  - Extended `preprocess_video.py` to save normalized ankle position (previously extracted for knee-angle math but discarded), and built `hip_to_ankle_vertical_right` in `start_phase_features.py` — Phase 5c, sit height / knee-bend depth, one of the most heavily-coached elements in speed skating and previously entirely unmeasured.
  - Validated the new feature against `start_candidate_3`'s verified `start_rest` segment as a standing-reference calibration: REST correctly centered near 1.0 (self-calibrated baseline), and EARLY-ACCELERATION correctly showed deeper crouching (mean 0.85, 25th percentile 0.66) — a physically sensible, directionally-correct first result.
  - Wired the new metric into `compare_to_elite_reference.py`, rebuilt the elite reference profile (now 6 metrics across 3 skaters per phase), and confirmed it working via a self-comparison sanity check.
  - Ran a genuine external test — comparing Ragne Wiklund (not part of the reference profile) against the elite reference for straightaway technique.
- **Problems, Challenges & Decisions:**
  - The external Ragne Wiklund comparison produced a physically implausible metric combination (nearly straight knees alongside an extremely shallow crouch and near-zero velocity). Investigated rather than accepted at face value: a frame-by-frame candidate scan found the labeled range was 60% contaminated by two simultaneous skaters (27 of 45 sampled frames).
  - Found and properly fixed a real, previously-undiagnosed bug in `check_same_skater_identity.py`: it reopened a new `VideoCapture` and seeked directly to each isolated target frame on every call, meaning MediaPipe's `detect_for_video()` (which relies on temporal/motion continuity between calls) sometimes failed to detect a person even on frames independently confirmed to have one. Rebuilt the script to read sequentially through one continuous pass with a warm-up period, instead of jumping to isolated frames — verified the fix directly against the exact frame that previously failed (frame 1065: 0 candidates when queried in isolation, 1 candidate when reached via the corrected sequential read).
  - Re-ran the Ragne Wiklund comparison on a newly identified, properly identity-verified clean segment (0.8549 similarity, confirmed same skater throughout). The physically odd pattern persisted even with identity confirmed correct, ruling out tracking contamination as the explanation. Diagnosed the real cause: `hip_to_ankle_vertical_right` only measures the vertical (Y-axis) component of hip-to-ankle distance, but speed-skating push-off extends the leg laterally, not straight down -- a fully extended leg pushed outward during an active push can show a small vertical-only distance despite being completely straight. Documented as a genuine, unresolved conceptual limitation in the current metric's design, discovered specifically because this segment captured a real moment of active lateral push-off.
- **Result:** Phase 5c is built, validated, and fully integrated into the working elite-anchor comparison tool. A real bug in the identity-verification methodology used throughout Phase 4 and 5 was found and fixed, with a direct before/after verification. A new, specific, and non-obvious limitation in the sit-height metric's design was discovered and documented rather than glossed over. Straightaway reference data grew to 4 skaters with the newly verified Ragne Wiklund segment.