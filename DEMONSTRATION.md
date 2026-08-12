# Biomechanics Unsupervised Anomaly Detection Pipeline

This repository hosts a machine learning and computer vision pipeline designed to ingest multivariate joint-angle time series, process movement patterns using a PyTorch autoencoder, and evaluate anomaly lead-time detection for sports science and biomechanics research.

---

## 🚀 Project Capabilities
* **Data Integration:** Processes structured multivariate time-series CSV datasets.
* **Deep Learning Inference:** Evaluates movement stability using a custom PyTorch autoencoder architecture via Mean Squared Error (MSE) reconstruction loss.
* **Lead-Time Evaluation:** Quantifies how early movement breakdown or anomalies are flagged prior to a critical threshold.
* **Visual Diagnostics:** Automatically exports analytical performance graphs to an `outputs/` directory.

---

## 📂 Repository Layout
The workspace strictly follows a modular layout to ensure robust absolute path resolution:
```text
biomechanics-project/
├── data/                       # Stores input CSV angle files
├── models/                     # Stores trained model weights (.pth)
├── outputs/                    # Stores generated evaluation plots
├── generate_mock_data.py       # Data generation and simulation script
├── calculate_lead_time.py      # Core deep learning evaluation script
└── requirements.txt            # Environment dependencies