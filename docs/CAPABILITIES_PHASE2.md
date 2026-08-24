# 🚀 Phase 2 Roadmap & Capabilities (`abilities_phase2.md`)

## 🟡 Part 2: Phase 2 Roadmap & In-Progress Development (Full Video Automation & Continuous Tracking)

* **🔄 End-to-End Continuous Video Pipeline (Core Phase 2 Objective):** Building an automated ingestion engine that takes an entire raw video file from start to finish, removing the need for manual segmenting and outputting a rolling fatigue timeline frame-by-frame.
* **⛸️ Automated Stride & Segment Extraction:** Developing and integrating peak-detection algorithms targeting hip, knee, and ankle coordinates to automatically parse continuous video streams into distinct stride cycles.
* **📊 Temporal Window Refactoring (`compare_research.py`):** Upgrading comparative research visualization logic to replace global indexing with explicit absolute frame and timestamp mapping, ensuring fresh vs. fatigued multi-panel curves accurately display true kinematic divergence.
* **👥 User-Independent Cross-Validation:** Training models on multi-skater datasets and validating generalization performance on completely unseen athletes to ensure robust, non-overfitted anomaly detection.
* **🔬 Advanced Sensitivity Analysis & Refined Ablation Studies:** Expanding systematic joint-removal experiments to mathematically identify which specific anatomical trajectory carries the highest predictive weight for fatigue anticipation.
* **🏆 ACSEF Competition Submission & Finalization:** Compiling all technical documentation, lead-time graphs, and experimental results into a presentation-ready format for high school science fair judging.
* **📏 Automated Data Normalization:** Implementing spatial scaling modules to normalize joint coordinates across different athletes, ensuring the autoencoder evaluates pure form rather than varying body proportions or camera distances.
* **🌟 Elite Benchmark Kinematic Template Directory:** Processing reference videos of world-class, professional speed skaters through the feature extractor to create a "gold-standard" baseline folder.
* **🖥️ Real-Time Edge Deployment (UI Integration):** Transitioning from offline video processing to a live web dashboard (Streamlit/OpenCV) capable of pulling live rinkside webcam frames and issuing real-time form warnings.
* **🌐 Cross-Discipline Adaptation:** Expanding the autoencoder's training baseline to handle both ice speed skating and inline speed skating simultaneously, evaluating how well the core double-push kinematic model generalizes across different friction surfaces.
* **📱 Lightweight Model Quantization:** Optimizing the PyTorch autoencoder weights for edge devices (such as NVIDIA Jetson or mobile hardware) to enable low-latency, on-device inference directly at the skating rink without heavy cloud computing dependencies.
