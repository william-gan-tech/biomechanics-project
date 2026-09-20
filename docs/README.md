# 🧊 Deep Learning Biomechanics & Injury Prevention Engine
*Advanced Temporal Trajectory Analysis for Speed Skating Form Breakdown, Fatigue Prediction, and Technique-Phase Biomechanics*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](docs/LICENSE)

> **Documentation integrity note:** On 9/12, this project underwent a full
> documentation audit after several previously-claimed results (a "100%
> variance reduction," working ONNX edge inference, multi-camera fusion
> status) could not be corroborated against the actual codebase. Those
> claims were retracted or downgraded. Since then, every phase's findings
> — including ambiguous or negative ones — have been independently
> verified before being written up. See `docs/FIXES.md` for the full
> correction record and `docs/PHASE1_SUMMARY.md` through
> `docs/PHASE4_SUMMARY.md` for the honest, evidence-backed result of each
> phase.

---

## 🎯 Formal Research Question

> **To what extent can deep learning models leverage comparative temporal joint-angle trajectories — across full sessions, fatigue states, and distinct technique phases (starts, corners, straightaways) — to proactively forecast biomechanical performance degradation and characterize technique differences in elite speed skaters, and to what extent does bone-length scaling improve this analysis across athletes of different builds?**

This umbrella question is answered incrementally, phase by phase — see **Core Project Status** below for what's actually been verified at each stage.

---

## A Little Self Introduction

> As someone deeply immersed in sports, AI, and robotics, I decided to build this model because I am a speed skater myself, having competed internationally and earned national gold medals representing Team USA, while also sharing a background in endurance sports like cross country, track, and running half marathons in San Jose and San Francisco. Between experiencing those sports firsthand and achieving a #11 world ranking in VEX robotics, I've always been fascinated by how technology and athletics intersect. I watched runners and skaters struggle with fatigue, noticing that humans usually only spot form breakdown — like skaters not bending their knees, or runners losing proper posture — after it has already happened and they are already slowing down. That sparked my core question: to what extent can deep learning models leverage temporal joint-angle trajectories to anticipate biomechanical performance degradation prior to measurable athletic deceleration in speed skaters? By combining my athletic background with my love for AI and robotics, I wanted to build something that moves past reactive observation into true predictive intelligence.

---

## 💡 Why This Project Matters

* **Proactive Injury Prevention:** Learns an individual skater's normal movement baseline and flags mechanical drift as it starts to occur.
* **Objective Coaching Intelligence:** Compares developing athletes against elite reference forms using real, computed kinematic data rather than guesswork.
* **Phase-Specific Technique Analysis:** Distinguishes start, corner, and straightaway mechanics — real, measured findings showing these phases have meaningfully different biomechanical signatures, not just "skating in general."
* **Early-Warning Capability:** Investigates whether temporal trajectory analysis can catch mechanical breakdown before it's visible as physical deceleration.

---

## ⚡ Core Project Status & Architecture

### Phase 1 — Completed
Proved deep learning autoencoders and LSTM architectures can distinguish fresh from fatigued movement on a single subject, before observable athletic deceleration.

### Phase 2 — Mostly Completed
Automated end-to-end video ingestion (`pipeline_engine.py`), a working Streamlit dashboard (`app.py`), and automated baseline calibration are functional and verified.

**ONNX edge acceleration — partially functional, corrected 9/16:**
- **FP32 export: genuinely functional.** Numerically equivalent to PyTorch (max difference 0.000031) and **measured 3.08x faster** (0.918ms vs. 2.824ms mean, batch size 8, CPU) via a real, reproducible benchmark (`benchmark_onnx_speed.py`).
- **INT8 quantization: file-size bug fixed, not usable.** Now genuinely smaller (144KB vs. 495KB, a real 70.9% reduction), but output verification shows a max difference of 55.7 vs. PyTorch — a genuine correctness bug, documented as unresolved rather than claimed as working.

### Phase 3 — Complete
> *To what extent can relative bone-length scaling improve cross-subject generalization in detecting neuromuscular fatigue across diverse athletes?*

Two real experiments (3a: general cross-subject variance; 3b: fatigue-detection separability specifically), reproduced results, proper paired statistical testing (Wilcoxon, all non-significant at n=7 — reported honestly), and a genuine outlier-sensitivity finding: **the headline result reverses when one camera-cutaway-affected skater is excluded** (variance increases with scaling at n=7, decreases at n=6 in 3 of 4 comparisons). Full methodology and both versions of the result in `docs/PHASE3_SUMMARY.md`.

### Phase 4 — Complete (9/16–9/18)
> *To what extent can bone-length-scaled trajectories distinguish start-phase acceleration from steady-state cruising, and does multi-person tracking correctly isolate one skater from simultaneous competitors?*

**4a (biomechanical):** Automated start detection failed on real Olympic footage (landmark jitter on chaotic broadcast video); pivoted to manual visual verification and found a genuine, confirmed start sequence. Real within-subject comparison across start/corner/straightaway phases found **torso-lean stability: straightaway (26.2° std) > start (35.9°) >> corner (82.9°)** — corners 3.2x more variable, matching known cornering biomechanics — plus real knee asymmetry in corners (14° gap) vs. near-symmetry on straightaways (2° gap).

**4b (tracking robustness):** Validated the 9/15 multi-person identity tracking system against genuine Olympic pack-racing footage — 135 real ambiguous multi-person events across the full video, with a 10-sample manual review confirming correct identity resolution every time.

Full methodology, including two caught-and-corrected false positives (broadcast graphic overlays mistaken for real acceleration spikes) and the identity-resolution process, in `docs/PHASE4_SUMMARY.md`.

### Phase 5 (planned) — Straightaway Stroke Mechanics
Already has a real preliminary baseline from Phase 4's technique-phase comparison (straightaway shown to be the most kinematically stable phase). Next: expand into a full three-condition (unscaled/scaled/zscore-only) ablation specific to straightaway segments.

### Phase 6 (planned) — Corner Technique
Already has concrete, evidence-backed motivation: Phase 4's measured corner-phase knee asymmetry (14° vs. 2°) demonstrates that the current right-side-only feature set is insufficient for this phase — bilateral (left+right) tracking is a real, demonstrated requirement, not a speculative addition.

---

## 📊 Verified Pipeline & Dashboard Capabilities

* **🎥 Automated Video Ingestion:** Processes raw MP4 files and YouTube URLs end-to-end via `yt_dlp`.
* **🤖 Real Pose Estimation:** MediaPipe Tasks `PoseLandmarker` API (migrated 9/08 off the removed legacy `mp.solutions.pose` API).
* **🦴 Bone-Length Calibration:** Auto-skips intro/title cards, picks whichever leg is visible per frame, and — as of 9/15 — maintains correct skater identity via position continuity and appearance-histogram matching when multiple people are in frame. Validated against genuine pack-racing footage 9/18 (135/135 sample-reviewed correct).
* **📐 Real Biomechanical Features:** Knee flexion angles, hip/shoulder position, torso-lean/crouch angle, hip velocity/acceleration (Phase 4 additions).
* **👥 Real Cross-Skater Comparison:** DTW-based comparison with honest tiered fallback (video → CSV → clearly-labeled simulated) instead of faking results when data's unavailable.
* **🏁 Real Fresh-vs-Fatigued Analysis:** Real stride-frequency and variability computation where source data exists, honest simulated fallback otherwise.
* **⚡ ONNX Edge Runtime:** FP32 genuinely functional and 3.08x faster (verified 9/16); INT8 documented as a known, unresolved correctness limitation.
* **🖥️ Interactive Dashboard:** Persistent session state, live thresholds, auto-digest mode, CSV export.
* **📊 Statistical Ablation Framework:** Leave-One-Skater-Out cross-validation, paired significance testing, incremental/resumable experiment runs.

---

## 🚀 Getting Started & Execution

```bash
# Install dependencies (requirements.txt is a real, verified pip freeze as of 9/12)
pip install -r requirements.txt

# Run the Streamlit dashboard
python -m streamlit run app.py

# Phase 3 research scripts
python -m run_bone_scaling_ablation           # 3a: general cross-subject variance
python -m run_fatigue_separability_ablation   # 3b: fatigue-detection separability
python -m analyze_phase3_results              # Combined statistical analysis
python -m outlier_sensitivity_check           # Outlier robustness check

# Phase 4 research scripts
python -m find_candidate_start_moments        # Automated start detection (documented limitations)
python -m browse_frames_manually --video <path> --start <s> --end <s>  # Manual verification fallback
python -m compare_technique_phases            # Start/corner/straightaway comparison
python -m diagnose_tracking_swap              # Multi-person tracking validation

# ONNX
python -m export_onnx_model     # Real export + quantization
python -m benchmark_onnx_speed  # Real measured speed comparison
```

Always run with Python's `-m` flag to avoid relative import path issues.

---

## 📈 Project Documentation

All tracking documents live in `docs/`:

* **`JOURNAL.md`**, **`HOURS.md`** — daily engineering log and time tracking, with a 9/12 correction pass on earlier entries
* **`CAPABILITIES_PHASE1.md`** through **`CAPABILITIES_PHASE4.md`** — verified capability logs per phase
* **`PHASE1_SUMMARY.md`** through **`PHASE4_SUMMARY.md`** — honest, narrative research conclusions per phase
* **`MILESTONES.md`** — high-level milestone summary
* **`FIXES.md`** — consolidated correction/retraction log
* **`DEMONSTRATION.md`** — setup and usage guide, with a correction banner on outdated screenshots

---

## 🛠️ System Architecture

```text
[Raw Video / YouTube URL] → [pipeline_engine.py: yt_dlp + MediaPipe PoseLandmarker
                              + multi-person identity tracking]
                                          │
              [Bone-Length Calibration + Butterworth Filtering
               + Start/Corner/Straightaway Feature Extraction]
                                          │
                    [PyTorch LSTM Autoencoder: Reconstruction Loss]
                                          │
        [Streamlit Dashboard: Metrics, Charts, CSV Export, Cross-Skater Comparison]
```

---

## 🧹 Repository Notes

Some `.onnx` model files reflect an attempted (and now partially real, partially documented-as-broken) quantization — see Phase 2 status above. Ephemeral cache/temp files are gitignored as of 9/12; legacy exploratory scripts are preserved in `archive/` rather than deleted. Git tags mark key historical milestones (`v1-real-cross-skater-comparison`, `v2-repo-cleanup-complete`, `v3-multiperson-tracking-fix`).