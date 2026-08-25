# 🚀 Phase 2 Roadmap & Capabilities (`abilities_phase2.md`)

## 🟡 Part 2: Phase 2 Roadmap & Completed/In-Progress Development (Full Video Automation & Continuous Tracking)

* **🔄 End-to-End Continuous Video Pipeline (Core Phase 2 Objective) - [COMPLETED]:** Built an automated ingestion engine (`pipeline_engine.py`) that processes raw video files from start to finish, removing manual segmenting and outputting a rolling fatigue timeline frame-by-frame.
* **🖥️ Real-Time Web Dashboard Integration (Streamlit UI) - [COMPLETED]:** Successfully integrated backend auto-digestion into an advanced Streamlit dashboard (`dashboard.py`), allowing users to upload MP4 trials and instantly generate dynamic thresholds, summary metrics, and anomaly tables.
* **📈 Automated Baseline Calibration - [COMPLETED]:** Implemented statistical mean and standard deviation calculations over initial window streams to establish dynamic, automated anomaly detection thresholds ($Mean + 2.0 \times Std$).
* **⛸️ Automated Stride & Segment Extraction - [IN PROGRESS]:** Developing and integrating peak-detection algorithms targeting knee and ankle flexion coordinates to automatically parse continuous video streams into distinct stride cycles.
* **📊 Temporal Window Refactoring (`compare_research.py`):** Upgrading comparative research visualization logic to replace global indexing with explicit absolute frame and timestamp mapping, ensuring fresh vs. fatigued multi-panel curves accurately display true kinematic divergence.
* **👥 User-Independent Cross-Validation:** Training models on multi-skater datasets and validating generalization performance on completely unseen athletes to ensure robust, non-overfitted anomaly detection.
* **🔬 Advanced Sensitivity Analysis & Refined Ablation Studies:** Expanding systematic joint-removal experiments to mathematically identify which specific anatomical trajectory carries the highest predictive weight for fatigue anticipation.
* **🏆 ACSEF Competition Submission & Finalization:** Compiling all technical documentation, lead-time graphs, and experimental results into a presentation-ready format for high school science fair judging.
* **📏 Automated Data Normalization:** Implementing spatial scaling modules to normalize joint coordinates across different athletes, ensuring the autoencoder evaluates pure form rather than varying body proportions or camera distances.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class, professional speed skaters through the feature extractor to create a "gold-standard" baseline folder.
* **📱 Lightweight Model Quantization:** Optimizing PyTorch autoencoder weights for edge devices (such as NVIDIA Jetson or mobile hardware) to enable low-latency, on-device inference directly at the skating rink without heavy cloud computing dependencies.
