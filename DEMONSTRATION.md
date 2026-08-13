# Biomechanics Project Master Setup & Requirements Guide

This document combines complete environmental setup instructions, software requirements, and demonstration guidelines so anyone can set up and run your project from scratch.

---

## Part 1: Initial Environment Setup & Prerequisites

### Step 1: Install Python 3.12
1. Go to python.org and download Python 3.12.
2. Run the installer `.exe` file.
3. ⚠️ **CRITICAL:** Check the box at the bottom that says **"Add python.exe to PATH"** before clicking **Install Now**.

### Step 2: Install Microsoft C++ Runtime
This installs missing system files like `c10.dll` so PyTorch loads properly without crashing.
1. Download the Microsoft Visual C++ Redistributable (x64) installer from Microsoft: [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Run `vc_redist.x64.exe` and follow the prompts. Restart your computer if Windows asks you to.

### Step 3: Set Up Project Folder & Terminal
1. Create a folder on your Desktop named `Research - biomechanics_project`.
2. Open Command Prompt (Press `Win + R`, type `cmd`, and hit Enter).
3. Navigate into your folder:
   ```dos
   cd "C:\Users\<YourUsername>\OneDrive\Desktop\Research - biomechanics_project"

---  

# Biomechanics Unsupervised Anomaly Detection Pipeline

This repository hosts a machine learning pipeline designed to ingest multivariate joint-angle time series, process movement patterns using a PyTorch autoencoder, and evaluate anomaly detection for biomechanics research.

---

## 🚀 Project Capabilities
* **Data Integration:** Processes structured multivariate time-series CSV datasets.
* **Deep Learning Inference:** Evaluates movement stability using a custom PyTorch autoencoder architecture via Mean Squared Error (MSE) reconstruction loss.
* **Anomaly Profiling:** Quantifies when movement breakdown or fatigue anomalies are flagged prior to a critical threshold.
* **Visual Diagnostics:** Automatically exports analytical performance graphs to an `outputs/` directory.

---

## Automated Setup & Demonstration
Copy and paste the following single command into your terminal to clone, install, generate data, and run the complete model demo instantly:

git clone [https://github.com/william-gan-tech/biomechanics-project.git](https://github.com/william-gan-tech/biomechanics-project.git) && cd biomechanics-project && pip install -r requirements.txt && python generate_mock_data.py && python demo_skater_a.py (starting from git clone and ending at demo_skater_a.py is the code you need to paste!)

After running the command above, check the outputs/demo_result.png file to view the generated model reconstruction error and fatigue threshold profile.

## 📂 Repository Layout & Quick Start Instructions

```text
biomechanics-project/
├── data/                       # Stores input CSV angle files
├── models/                     # Stores trained model weights (.pth)
├── outputs/                    # Stores generated evaluation plots
├── generate_mock_data.py       # Data generation and simulation script
├── demo_skater_a.py            # Single-subject demonstration script
└── requirements.txt            # Environment dependencies


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
