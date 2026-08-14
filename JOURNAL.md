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
 
Created a README.md to give an introduction of my research project for others.

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

- Made an hours log to keep track of key progress points and time spent
- Started a file for capabilities of my research project currently and what I plan for it to do in the future (different phases).

💡 Core Engineering & Research Decisions:

  1. Shifting from Post-Hoc Analysis to Proactive Prevention: Decided to scale the autoencoder framework from a purely diagnostic data extractor into an active warning ecosystem. This ensures the system does not just look backward at past performance, but serves as a real-time safeguard to prevent overuse injuries and structural breakdown.

  2. Cross-Discipline Generalization (Ice & Inline Skating): Unified the project's mechanical kinematics to cover both ice speed skating and inline speed skating, recognizing that their fundamental double-push     biomechanics and cornering profiles share identical underlying spatial data.

  3. Establishing Rigorous Documentation Standards: Committed to maintaining open-source transparency on GitHub via structured capability logs and time audits, ensuring the project's evolution is traceable, reproducible, and ready for formal research evaluation.

(Important): Made a section in my README.md for why my research project matters and what real world impact it can have.

## 8/10: Local Workspace Migration, Repository Structure & Pipeline Debugging
- **Action Taken:**
  - Migrated core project artifacts from Google Colab into a local **VS Code** development environment to establish an offline, modular workflow.
  - Reorganized the local workspace into a clean scientific directory structure consisting of dedicated `data/`, `models/`, and `outputs/` subfolders.
  - Successfully initialized Git source control tracking locally, committing code changes, setting up the `requirements.txt` dependency file, and pushing updates to the remote GitHub repository (`biomechanics-project`) to maintain a verifiable engineering timeline.
  - Attempted execution of the comparative lead-time evaluation script (`calculate_lead_time.py`) and diagnosed data path configurations.
  - Performed data verification tests (`check_data.py` and `preprocess_video.py`) confirming the integrity of multi-joint coordinate extractions across frame sequences.

- **Problems, Challenges & Decisions:**
  1. **Directory Path Resolution & File Structure Organization:**
     - *The Problem:* Initially, nested folder creation resulted in a combined directory path (`data\output\models`), and subsequent script executions threw `FileNotFoundError` exceptions because Python expected files inside specific subpaths.
     - *The Decision:* Cleaned up the root directory by deleting the malformed paths and establishing three independent root-level directories (`data/`, `models/`, `outputs/`). Updated relative read paths in local scripts to cleanly interface with the `data/` asset folder without breaking code execution.
  2. **Handling Video Extraction Bottlenecks:**
     - *The Problem:* Encountered limitations during secondary skater video extraction due to pose landmarker detection tolerances in raw footage frames.
     - *The Decision:* Documented the failure as an iterative engineering hurdle. Rather than halting development, decided to isolate preprocessing scripts and focus on finalizing the core anomaly-detection evaluation logic using validated multivariate baseline arrays first.
    
     During workspace configuration on August 10, setting up the project locally in VS Code introduced initial directory path resolution issues. Typing the folder destination all on one line accidentally merged the folder structure into a single nested path rather than creating independent sibling directories. Consequently, when running evaluation scripts, the relative path configurations in Python triggered a FileNotFoundError because the program looked for data files inside an unseparated directory. This was resolved by re-engineering the workspace root architecture, systematically deleting the malformed path structure, and establishing three isolated root directories to mirror standard machine learning layouts with a strict separation of concerns for assets, models, outputs, and scripts.

Additionally, attempts to process secondary skater video footage encountered computer vision tracking constraints. The pose landmarker framework experienced confidence drops and tracking loss due to rapid motion blur, suboptimal environmental lighting, and limb truncation as the skater moved out of frame boundaries. Rather than letting a hardware-level tracking hurdle stall pipeline progression, this extraction bottleneck was documented as an iterative engineering constraint. The preprocessing script was isolated to prioritize the core anomaly-detection evaluation logic using pre-validated baseline multivariate arrays, maintaining full momentum on the primary research goal while preserving a transparent log for technical evaluation.

## Goals: Fixing limitations to future skater video extractions; Data Pipeline & Synthetic Generation Fix
- **Plans:**
- **Workspace & Repository Management:** Migrated project files to a clean, local VS Code environment with a structured three-folder layout (`data/`, `models/`, `outputs/`) and synchronized version control updates (`requirements.txt` and code commits) to GitHub.
- **Directory Path & Pipeline Troubleshooting:** Resolved `FileNotFoundError` exceptions by correcting relative path read configurations and cleaning up malformed path nesting in the project root.
- **Video Preprocessing & Tracking Analysis:** Documented computer vision tracking hurdles during secondary video extraction caused by environmental factors, motion blur, and frame boundary limits.
- **Strategic Iterative Pivot:** Isolated the preprocessing script to prioritize finalizing the core unsupervised anomaly-detection evaluation logic (`calculate_lead_time.py`) using pre-validated baseline multivariate arrays.

- **Issue:** Encountered intermittent `FileNotFoundError` exceptions when running data generation and model scripts. 
- **Root Cause:** Inconsistent relative path evaluation across execution threads and potential directory context shifts within the local workspace environment.
- **Solution:** 1. Transitioned all scripts to use robust **absolute path strings** (`os.path.join`).
  2. Implemented pre-flight existence checks (`os.path.exists`) in `generate_mock_data.py` to catch missing base assets immediately with descriptive errors instead of silent failures.
  3. Verified correct placement and naming conventions for baseline multivariate angle CSVs within the `data/` subdirectory.

Resolving FileNotFoundError & Absolute Path Resolution:

The Problem: During initial local workspace execution in VS Code, scripts experienced persistent FileNotFoundError exceptions when attempting to read baseline CSV datasets. This was caused by environment-specific path virtualization and workspace context mismatches where execution threads failed to resolve relative paths correctly, or automated folder creation nested directories into unintended structures (e.g., data/output/models).

The Decision: Refactored the codebase to eliminate fragile relative path referencing. Transitioned all scripts to use robust absolute path routing via os.path.join and built-in pre-flight assertions (os.path.exists) to catch missing assets immediately with clear descriptive errors instead of silent failures. Systematically cleaned the root directory to maintain three isolated root-level sibling folders (data/, models/, outputs/) to mirror standard machine learning architectures.

During the setup of the biomechanics research pipeline, the project encountered a persistent FileNotFoundError when attempting to load baseline CSV datasets. This issue stemmed from environment-specific path virtualization and workspace context mismatches where Python's execution thread failed to resolve relative paths or synced cloud directories correctly. To resolve this, the codebase was refactored to implement strict absolute path routing using os.path utilities alongside pre-flight file existence assertions (os.path.exists()). Furthermore, a programmatic file picker fallback using tkinter was introduced to bypass automated path resolution errors entirely when running scripts across different local machine configurations.

I've decided to focus on other parts of this project to make progress for now and come back to these two problems later (because the 'FileNotFoundError' isn't fully resolved for my model yet). I plan to find a way to make it possible for my AI to be able to analyze different sorts of videos and clip (even if they aren't the best quality!) and figure/learn more about Python and VScode to continue on my coding work for this research project.
 
## 8/11: Single-Subject Demo Refinement, Documentation & Repository Finalization
Action Taken:

Single-Subject Demo Development (demo_skater_a.py): Successfully diagnosed multi-skater pipeline bottlenecks, pivoted strategy, and built a clean, self-contained standalone demo script focused exclusively on Skater A (extracted_multivariate_angles.csv and extracted_knee_angles.csv).

Automated Visual Outputs: Configured the demonstration script to dynamically load the angle data, execute model anomaly/reconstruction error calculations, simulate a fatigue threshold, and automatically export a visual performance graph (outputs/demo_result.png).

Documentation Architecture (DEMONSTRATION.md & REQUIREMENTS.md): Developed and refined professional, user-friendly markdown documentation outlining project capabilities, workspace layouts, system prerequisites, and a one-line automated setup/execution command.

Version Control & GitHub Synchronization: Staged, committed, and pushed all updated code artifacts, requirements lists, and demo instructions to the remote GitHub repository (biomechanics-project).

- **Problems, Challenges & Decisions:**

Simplifying Multi-File Pipeline Obstacles: * The Problem: Attempting to run a simultaneous multi-skater comparative demo (Skater A & B) introduced complex path or data synchronization issues that stalled immediate progress.

- **The Decision:** Strategically pivoted to a simplified, single-subject demonstration framework centered on Skater A. This isolates the core autoencoder anomaly logic cleanly, allowing anyone reviewing the repository to instantly execute and visualize the model's output without friction.

Streamlining Onboarding & Reproducibility: * The Problem: Reviewers or peers trying to test machine learning models often struggle with fragmented instructions and manual package installations.

The Decision: Streamlined the setup workflow into a single-line terminal command (git clone ... && pip install ... && python ...) combined with explicit Python 3.12 prerequisites and a clean requirements.txt manifest, ensuring total reproducibility on any local machine.

- **Goals & Plans for the Future:**
- 
Refining Multi-Video Scalability: Return to the secondary skater video extraction pipeline to handle lower-quality or unconstrained footage tracking hurdles.

Interactive UI Integration: Explore packaging the model results into an interactive web dashboard (such as Streamlit) for coaches and researchers to view real-time feedback.


## 8/12: Interactive Dashboard Deployment & Multi-Joint Architecture Expansion
Action Taken:

Successfully built, tested, and locally deployed an interactive web application (app.py) powered by Streamlit, translating raw backend PyTorch anomaly scores into a functional data-science dashboard.

Configured real-time dashboard controls including dynamic anomaly threshold sliders, metric calculation summaries, time-series reconstruction error trend lines, and downloadable CSV summary reporting.

Synchronized repository changes, pushing the complete local pipeline (main.py, app.py, requirements.txt, and data artifacts) to the public GitHub repository (william-gan-tech/biomechanics-project).

- **Problems, Challenges & Decisions:**

Resolving Local Server Interruption & Blank Browser States:

- **The Problem:** When executing python -m streamlit run app.py, the local server occasionally dropped or rendered a blank screen because of unexpected command inputs or missing fatigue_results.csv data assets.

- **The Decision:** Implemented pre-flight data validation checks within app.py using os.path.exists() to verify that the core anomaly results file is present before attempting to render graphs, preventing silent rendering failures.

Bridging Backend Data Extraction to Frontend UI:

- **The Problem:** Transitioning a machine learning model's command-line output into an intuitive interface requires structured data serialization that Streamlit can process efficiently.

- **The Decision:** Standardized the CSV schema export format in main.py to strictly use a two-column index-to-score structure (Window_Index, Anomaly_Score), ensuring seamless compatibility with Streamlit's native dataframes and line charts.

- **🚀 Future Development Roadmap (Updated)**

🌐 Cloud Deployment (Streamlit Community Cloud): Finalizing public hosting setup via share.streamlit.io to transition the local localhost:8501 interface into a publicly accessible URL for peer review and research presentation.

🔍 Joint-Specific Error Decomposition: Upgrading the PyTorch autoencoder's loss function to preserve individual feature channels (reduction='none'), allowing the Streamlit dashboard to feature multi-joint toggle views (e.g., isolating knee flexion vs. hip alignment).

⏱️ Quantitative Lead-Time Analysis: Expanding predictive scripts to calculate the exact frame and second advantage the autoencoder provides prior to observable athletic deceleration.
