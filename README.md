# Anticipating Speed Skater Performance Degradation via Temporal Joint Trajectories

## 📌 Project Abstract
Can deep learning models leverage temporal joint-angle trajectories to anticipate biomechanical performance degradation prior to measurable athletic deceleration in speed skaters? 

This research focuses on extracting 2D/3D kinematic joint coordinates (hip, knee, ankle) from video footage using MediaPipe Pose, smoothing joint motion using low-pass signal filtering (Butterworth), and training temporal sequential models (LSTM/TCN) to detect micro-kinematic "fatigue signatures" before visible drops in skater velocity occur.

---

## 🛠️ Environment Setup & Installation

To replicate this environment locally, clone the repository and run the following commands:

```bash
# 1. Install PyTorch (ML Framework)
py -3.12 -m pip install torch torchvision torchaudio

# 2. Install OpenCV (Computer Vision Framework)
py -3.12 -m pip install opencv-python

# 3. Install Compatible MediaPipe (Pose Tracking Module)
py -3.12 -m pip install "mediapipe<0.10.14" --force-reinstall

# 4. Install Data Manipulation & Signal Processing Libraries
py -3.12 -m pip install numpy pandas matplotlib scipy scikit-learn

---

## 📊 Progress Tracking

Track live project updates, upcoming milestones, and sprint progress on the official project board:

👉 **[View GitHub Project Board](https://github.com/users/william-gan-tech/projects/1)** *(Replace URL with your exact board link)*

---

## 🚀 How to Run the Verification Script

Execute the main script to verify PyTorch and MediaPipe pose tracking:

```bash
py -3.12 main.py

4. Press **`Ctrl + S`** to save your `README.md` file.

---

### Step 2: Send It Up to GitHub

Now, in your VS Code terminal (or Command Prompt), run these two commands to publish your complete README online:

```cmd
git commit -am "Updated README with complete project documentation"

git push