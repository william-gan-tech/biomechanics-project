biomechanics-project/
│
├── main.py                # Core PyTorch model training & joint error decomposition
├── app.py                 # Interactive Streamlit web dashboard with dynamic thresholding
├── dashboard.py           # Polished main Streamlit dashboard with ONNX & Phase 2 export
├── pipeline_engine.py     # Automated end-to-end video processing backend
├── skating_model.onnx     # Optimized ONNX edge runtime model weights
├── data_loader.py         # Video ingestion & MediaPipe processing
├── biomechanics_utils.py  # Butterworth filter & angle math calculations
├── src/                   # Advanced processing scripts (batch eval, stress test, heatmaps)
├── Docs/                  # Visual artifacts and documentation (including joint error heatmaps)
├── data/                  # Raw and processed multivariate CSV datasets (elite speed skating trials)
├── models/                # Saved PyTorch autoencoder weights (.pth)
├── outputs/               # Consolidated multi-file batch execution telemetry & reports
├── requirements.txt       # Explicit python dependency tracking
├── HOURS.md               # Quantitative time and development log
├── JOURNAL.md             # Engineering thought process & milestones
├── abilities_phase1.md    # Phase 1 completed capabilities and milestone log
└── abilities_phase2.md    # Phase 2 active objectives and automation roadmap update