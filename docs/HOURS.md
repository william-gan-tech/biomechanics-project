| Date | Development Phase / Task Description | Hours Spent |
| :--- | :--- | :--- |
| **8/06** | Environment setup, Python 3.12 installation, runtime configuration, and initial script verification. | 1.0 hr |
| **8/07** | VS Code workspace setup and GitHub repository initialization (`biomechanics-project`) for version control. | 2.0 hrs |
| **8/08** | Integrated model pipeline, executed sliding window segmentation, trained PyTorch CUDA autoencoder, exported anomaly scores, and created introductory README.md. | 3.0 hrs |
| **8/09** | Developed strategic roadmap, injury-prevention framework, and project documentation (hours log, capability files, impact section). | 3.0 hrs |
| **8/10** | Migrated local workspace to VS Code, resolved path resolution bugs (`FileNotFoundError`), and refactored scripts for absolute paths. | 3.0 hrs |
| **8/11** | Built single-subject standalone demo (`demo_skater_a.py`), generated automated visualization outputs, and created markdown documentation. | 2.0 hrs |
| **8/12** | Deployed interactive Streamlit web dashboard (`app.py`) featuring dynamic threshold sliders, metric cards, and CSV export reporting. | 1.0 hr |
| **8/13** | Implemented dynamic statistical thresholding (mean plus two standard deviations) and joint-specific reconstruction error decomposition. Resolved GitHub push conflicts and updated project documentation. | 2.0 hrs |
| **8/14** | Executed synthetic failure stress-testing (`stress_test.py`), configured automated batch multi-file export (`batch_export.py`), and generated time-series multi-joint error heatmaps (`generate_heatmap.py`). | 2.0 hrs |
| **8/15** | Conducted predictive lead-time experiments, resolved MediaPipe C-binding and pathing bugs, and advanced Phase 1 analysis and multi-video ingestion testing. | 3.0 hrs |
| **8/16** | Replaced `moviepy` with OpenCV for video trimming, built ablation and evaluation frameworks, integrated `SkatingDegradationLSTM`, and started debugging fresh vs. fatigued visualization indexing. | 4.0 hrs |
| **8/17** | Optimized pre-deceleration label distributions, tested binary classification thresholds, and documented visualization debugging roadmap. | 1.0 hr |
| **8/18** | Ingested long-form 6-minute time trial video (`skater_time_trial.mp4`), implemented absolute frame mapping for fresh vs. fatigued states, and successfully resolved visualization indexing bugs in `compare_research.py`. | 2.0 hrs |
| **8/19** | Processed Mia Manganello Kilburg's footage, standardized multi-subject file names, and updated the Streamlit dashboard for dynamic skater selection. | 3.0 hrs |
| **8/20** | Added dataset paths, integrated 3000m trial data, attempted `yt-dlp` scraping for Haralds Silovs, and resolved format extraction errors by pivoting to native YouTube embeds. | 3.0 hrs |
| **8/22** | Finalized Phase 1 milestone, structured Phase 2 automation objectives, and updated project capability logs and development journals. | 2.5 hrs |
| **8/23** | Built automated video ingestion engine (`pipeline_engine.py`) and integrated live video auto-digestion into the Streamlit UI (`dashboard.py`). | 2.5 hrs |
| **8/24** | Integrated YouTube URL ingestion support into the pipeline, deployed adaptive stream fallback strategies using `yt-dlp`, debugged FFmpeg dependency and format-restriction blockers on specific YouTube IDs, and updated Phase 2 capability roadmap docs. | 1.0 hr |
| **8/25** | Integrated automated baseline calibration (`calibrate_baseline`), verified end-to-end dashboard telemetry, documented third-party streaming constraints (FFmpeg/format limits) as a core engineering challenge and obstacle for this project so far. | 2.0 hrs |
| **8/26** | Implemented edge optimization, PyTorch model quantization (FP32 to INT8), ONNX runtime deployment (`skating_model.onnx`), and finalized Phase 2 UI polishing and reporting features. | 2.5 hrs |
| **8/27** | Refactored Streamlit dashboard UI state architecture using persistent session state management (`st.session_state`), resolved widget rerun rendering bugs, synchronized Phase 2 capability logs, and stabilized frontend component execution. | 2.5 hrs |
| **8/28** | Standardized execution using Python's `-m` module flag, finalized Phase 2 completion, and synchronized documentation for ACSEF presentation. | 2.0 hrs |
| **8/29** | Conducted exhaustive Phase 1 and Phase 2 documentation sprint, synthesized markdown architecture files (`abilities_phase1.md`, `capabilities_phase2.md`), integrated personal athletic and robotics background into project narrative, and mapped end-to-end telemetry flows. | 1.5 hrs |
| **8/30** | Resolved Mode 5 Streamlit state-handling bugs and downstream `NameError` exceptions, synchronized project documentation across all tracking files, and established the Phase 3 roadmap for multi-athlete generalized validation and multi-view camera fusion. | 1.0 hr |
| **8/31** | Resolved schema feature validation errors in `pipeline_engine.py` by aligning checks to the 6 normalized filter features, and verified live stream ingestion resilience using external references like [YouTube Video](https://www.youtube.com/watch?v=06TE_U21FK4&t=1s). | 2.0 hrs |
| **9/01** | Stabilized pipeline validation rules, aligned 6-feature multivariate inputs, verified end-to-end Streamlit YouTube streaming ingestion workflow, and documented stabilization updates. | 1.5 hrs |
| **9/02** | Developed Leave-One-Subject-Out (LOSO) evaluation framework (`evaluate_ablation.py`), verified quantitative MSE drop (~4,500 to ~0.62) via bone-length/feature scaling, updated dataset normalization logic, and fixed dashboard `WinError 32` file locks. | 2.0 hrs |
| **9/03** | Developed multi-angle stream synchronization module (`fusion_engine.py`), exported PyTorch autoencoder to ONNX edge runtime (`skating_model.onnx`) with dynamic axes, and integrated asynchronous threaded downloading for live YouTube ingestion. | 1.0 hr |
| **Total** | **Cumulative Engineering Time** | **59.0 hrs** |
