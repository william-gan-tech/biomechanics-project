# Biomechanics Unsupervised Anomaly Detection Pipeline

This repository hosts a machine learning pipeline designed to ingest multivariate joint-angle time series, process movement patterns using a PyTorch autoencoder, and evaluate anomaly detection for biomechanics research.

---

## 🚀 Project Capabilities
* **Data Integration:** Processes structured multivariate time-series CSV datasets.
* **Deep Learning Inference:** Evaluates movement stability using a custom PyTorch autoencoder architecture via Mean Squared Error (MSE) reconstruction loss.
* **Anomaly Profiling:** Quantifies when movement breakdown or fatigue anomalies are flagged prior to a critical threshold.
* **Visual Diagnostics:** Automatically exports analytical performance graphs to an `outputs/` directory.

---

## 📂 Repository Layout & Quick Start Instructions

```text
biomechanics-project/
├── data/                       # Stores input CSV angle files
├── models/                     # Stores trained model weights (.pth)
├── outputs/                    # Stores generated evaluation plots
├── generate_mock_data.py       # Data generation and simulation script
├── demo_skater_a.py            # Single-subject demonstration script
└── requirements.txt            # Environment dependencies

## Automated Setup & Demonstration
Copy and paste the following single command into your terminal to clone, install, generate data, and run the complete model demo instantly:

git clone [https://github.com/william-gan-tech/biomechanics-project.git](https://github.com/william-gan-tech/biomechanics-project.git) && cd biomechanics-project && pip install -r requirements.txt && python generate_mock_data.py && python demo_skater_a.py

After running the command above, check the outputs/demo_result.png file to view the generated model reconstruction error and fatigue threshold profile.

## 📂 Repository Layout
The workspace strictly follows a modular layout to ensure robust absolute path resolution:
```text
biomechanics-project/
├── data/                       # Stores input CSV angle files
├── models/                     # Stores trained model weights (.pth)
├── outputs/                    # Stores generated evaluation plots
├── generate_mock_data.py       # Data generation and simulation script
├── demo_skater_a.py            # Single-subject demonstration script
└── requirements.txt            # Environment dependencies
