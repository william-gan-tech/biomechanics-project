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

## Goals: Fixing limitations to future skater video extractions
- **Plans:**
- **Workspace & Repository Management:** Migrated project files to a clean, local VS Code environment with a structured three-folder layout (`data/`, `models/`, `outputs/`) and synchronized version control updates (`requirements.txt` and code commits) to GitHub.
- **Directory Path & Pipeline Troubleshooting:** Resolved `FileNotFoundError` exceptions by correcting relative path read configurations and cleaning up malformed path nesting in the project root.
- **Video Preprocessing & Tracking Analysis:** Documented computer vision tracking hurdles during secondary video extraction caused by environmental factors, motion blur, and frame boundary limits.
- **Strategic Iterative Pivot:** Isolated the preprocessing script to prioritize finalizing the core unsupervised anomaly-detection evaluation logic (`calculate_lead_time.py`) using pre-validated baseline multivariate arrays.
 
