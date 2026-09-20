# Repository Structure

## Root — active pipeline code

### Core pipeline
- `app.py` — main Streamlit dashboard (moved here from `src/` on 9/12; this is the live entry point — confirm this matches your Streamlit Cloud "main file path" setting)
- `pipeline_engine.py` — core video ingestion, bone-length calibration, multi-person identity tracking (added 9/15), feature extraction
- `preprocess_video.py` — MediaPipe pose extraction with bone-length normalization
- `cross_skater_compare.py` — real DTW-based cross-skater comparison (Dashboard Mode 1)

### Phase 3 (cross-subject generalization)
- `run_bone_scaling_ablation.py` — Phase 3a: cross-subject general-variance ablation
- `run_fatigue_separability_ablation.py` — Phase 3b: fatigue-separability ablation
- `analyze_phase3_results.py` — combined statistical analysis (paired Wilcoxon tests)
- `outlier_sensitivity_check.py` — outlier robustness check on 3a/3b results

### Phase 4 (start/corner/straightaway analysis)
- `audit_footage_for_starts.py` — checks existing footage for genuine simultaneous multi-person content
- `start_phase_features.py` — torso-lean angle, hip velocity/acceleration feature engineering
- `find_candidate_start_moments.py` — automated acceleration-spike start detection (documented limitations — see `CAPABILITIES_PHASE4.md`)
- `test_start_candidate.py` — general-purpose download + time-windowed candidate search
- `browse_frames_manually.py` — manual frame-by-frame visual verification (fallback when automated detection fails)
- `compare_start_vs_cruise.py` — early start-vs-cruise comparison (superseded by `compare_technique_phases.py`)
- `check_same_skater_identity.py` — appearance-histogram-based identity check
- `check_gap_continuity.py` — spatial-continuity-based identity check (more decisive than appearance alone)
- `compare_technique_phases.py` — real within-subject start/corner/straightaway comparison
- `diagnose_tracking_swap.py` — frame-by-frame multi-person tracking diagnostic (9/15, extended 9/18)

### ONNX edge acceleration
- `export_onnx_model.py` — real model export + quantization (9/16)
- `onnx_inference.py` — real `ONNXFatigueDetector` inference wrapper, actually calls `onnxruntime.InferenceSession`
- `benchmark_onnx_speed.py` — real measured PyTorch vs. ONNX speed comparison

### Diagnostics & utilities
- `diagnose_bone_scaling.py`, `diagnose_mia_outlier.py` — earlier calibration diagnostics
- `download_cross_skater_videos.py`, `download_and_check_start_video.py` — batch video downloaders (yt-dlp based)
- `try_bone_scaling.py`, `test_pipeline.py` — standalone quick-test scripts

## Supporting modules (verification status — see `docs/CAPABILITIES_PHASE3.md`)
- `model.py` — `SkatingLSTMAutoencoder` architecture — ✅ real, actively used by all ablation scripts
- `normalize_pose.py` — ✅ real, used
- `multi_view_fusion.py` — real interpolation/merge logic, **not yet integrated** into `pipeline_engine.py`, one known bug (row-position vs. true time-based interpolation)
- `cross_subject_normalization.py`, `dataset.py`, `evaluate_ablation.py`, `evaluate_generalization.py` — legacy Phase 3 exploration; `evaluate_generalization.py` specifically contains a confirmed bug (self-recursive `main()`) and does not test cross-subject generalization as originally claimed — see retraction in `docs/CAPABILITIES_PHASE3.md`

## `docs/`
- `JOURNAL.md`, `HOURS.md` — daily log and time tracking
- `CAPABILITIES_PHASE1.md` through `CAPABILITIES_PHASE4.md` — verified per-phase capability logs
- `PHASE1_SUMMARY.md` through `PHASE4_SUMMARY.md` — honest narrative research conclusions per phase
- `MILESTONES.md` — high-level milestone summary
- `FIXES.md` — consolidated correction/retraction log
- `DEMONSTRATION.md` — setup and usage guide
- `README.md` — kept in sync with root `README.md`
- `REQUIREMENTS.md` — real, verified `pip freeze` output (regenerated 9/12, grown since)
- this file, `LICENSE`

## `archive/`
- `src_legacy/` — stale duplicate copies of `dashboard.py`, `pipeline_engine.py`, `try_bone_scaling.py` found during the 9/12 cleanup (superseded by root-level versions)
- `phase1_2_scripts/` — ~60 exploratory/debugging scripts from Phase 1–2 development, preserved for history but not part of the active codebase
- `8-30-2026_documentation.md` — retired duplicate of `DEMONSTRATION.md` containing the same fabricated placeholder metrics; kept for record, not active documentation

## Data & output directories
- `data/`, `videos/` — source video files, including Phase 4 downloads (`start_candidate_*.mp4`) — most gitignored except intentionally committed samples
- `feature_cache/`, `ablation_feature_cache/` — cached extracted features, keyed by video modification time, regenerated automatically if stale
- `ablation_results/`, `fatigue_separability_results/` — experiment output CSVs and plots (Phase 3a/3b)
- `manual_frame_browse/`, `start_moment_candidates/` — Phase 4 visual-verification frame images
- `phase3_statistical_summary.md`, `phase4a_technique_phase_comparison.csv` (repo root) — auto-generated result files

## Known repo hygiene notes
- `skating_model.onnx` (FP32, genuinely functional as of 9/16), `skating_model_int8.onnx` (file-size bug fixed, but a real correctness bug remains — not usable for inference) — see `docs/CAPABILITIES_PHASE2.md`.
- `temp_primary_*.mp4`, `temp_skater_*.mp4`, `downloaded_skater.*`, `rendered_skating_output*.mp4` are gitignored as of 9/12 but may still exist in git history from before that date.
- `data/start_candidate_*.mp4` (Phase 4 downloads) — consider gitignoring if not already, same reasoning as other large source videos.