# Fixes & Corrections Log

This file consolidates every verified bug, fabrication, or unverifiable
claim found during the 9/12–9/15 documentation and code audit, along with
what was actually done about each. See individual phase capability docs
(`CAPABILITIES_PHASE1.md`, `CAPABILITIES_PHASE2.md`, `CAPABILITIES_PHASE3.md`)
for full technical detail — this file is the summary index.

Historical snapshots of the codebase at key points are preserved as git
tags rather than duplicated files — see **Git Tags** section near the
bottom.

---

## Code Bugs (real, fixed)

| Issue | Found | Fix |
|---|---|---|
| `mp.solutions.pose` API removed in installed MediaPipe version, breaking calibration silently | 9/08 | Migrated to `mediapipe.tasks.python.vision.PoseLandmarker` |
| `ROOT_DIR` pointed one directory above actual project root | 9/08 | Fixed path calculation |
| `sys.path` ordering let a stale `src/preprocess_video.py` shadow the correct file | 9/08 | Fixed import order |
| Calibration silently failed (returned hardcoded 0.45) on footage with intro/title cards | 9/08 | Added forward-scan to skip past non-subject frames |
| Ragne Wiklund's calibration silently failed due to one-leg occlusion | 9/11–9/12 | Bone-length calc now picks whichever leg is visible per frame |
| Mia Manganello Kilburg's footage had a mid-clip broadcast cutaway producing implausible landmark values | 9/11–9/12 | Added frame-level plausibility filter, applied to all skaters/conditions |
| Cross-Skater comparison mode used `np.random` fake data | 9/09 | Rebuilt with real DTW-based comparison (`cross_skater_compare.py`) |
| Anomaly threshold slider miscalibrated for the new comparison metric's scale | 9/09 | Added a dedicated, correctly-scaled threshold |
| Mode 2 (3000m fresh/fatigued) used hash-based fake numbers regardless of real data availability | 9/15-adjacent | Rewired to use real data where available (Patrick Meek, Mia), honest simulated fallback otherwise |
| Bone-scaling could silently swap to a different skater mid-video in multi-person footage | 9/15 | Added position-continuity + appearance-histogram identity tracking, EMA smoothing |
| `run_fatigue_separability_ablation.py` import order caused `ModuleNotFoundError` | 9/13 | Reordered imports |
| Original ablation script only saved results at the very end, losing ~3 hours of progress on interruption | 9/11 | Rebuilt to save after every fold, resumable |
| Live Streamlit Cloud deployment pointing at `src/dashboard.py` (archived, no longer exists at that path) | 9/15 | Identified as a genuinely broken deployment (not a "past version" — Streamlit Cloud runs live off current branch content, not snapshots) |

---

## Unverifiable / Fabricated Claims (retracted or downgraded, not deleted from history)

| Claim | Status | Evidence |
|---|---|---|
| "100.00% variance reduction, MSE 267.35–451.29 vs 0.47–0.58" | **RETRACTED** | Source script (`evaluate_generalization.py`) tests 4 arbitrary time-chunks of one video, not distinct subjects; contains a self-recursive `main()` that prevents a clean run |
| "ONNX edge runtime... lower CPU/GPU latency" | **DOWNGRADED to attempted/non-functional** | `onnxruntime` imported but never called; "int8" file (626KB) is larger than original FP32 (133KB) with mixed tensor types |
| "Multi-Angle Camera Stream Fusion... operational" | **DOWNGRADED, then reviewed** | File is `multi_view_fusion.py` (not `fusion_engine.py` as originally logged); contains real, sensible interpolation/merge logic but has one known bug (row-position interpolation instead of true time-based interpolation) and is not integrated into the live pipeline |
| "Asynchronous Multi-Threaded Streaming" | **UNVERIFIED** | No threading code found in `pipeline_engine.py` via keyword search |
| Mode 1–4 dashboard screenshots/metrics (91.4% Cross-Subject Accuracy, hardcoded Core Stability Index, etc.) | **RETRACTED for Modes 2–4, FIXED for Mode 1** | Confirmed hardcoded/seeded-random values in original source code, not computed measurements |
| `requirements.txt` package versions (e.g. `opencv-python==5.0.0.93`) | **REGENERATED, but note below** | This specific version number was later confirmed *real* when checked against the actual live environment — not every implausible-looking number turns out to be fabricated. Regenerated from a real `pip freeze` regardless, since trusting an AI-authored version list is the wrong process even when it happens to be right. |

---

## Repository Hygiene

- Committed ephemeral `temp_*.mp4`, `downloaded_skater.*`, `rendered_skating_output*.mp4` files untracked and gitignored (9/12)
- Nested duplicate `biomechanics-project/` folder (leftover from a clone-inside-a-clone) removed (9/12)
- ~60 exploratory Phase 1–2 scripts and 3 stale duplicate files in `src/` (`dashboard.py`, `pipeline_engine.py`, `try_bone_scaling.py`) archived to `archive/` rather than deleted, preserving history without cluttering the active codebase (9/12) — these are outdated *code*, not documentation claims; kept because they were real, functioning (if buggy/outdated) parts of the project at the time
- `ablation_feature_cache/`, `fatigue_separability_results/`, `ablation_results/` added to `.gitignore` — regeneratable experiment cache/output, not source (9/15)
- Recurring git divergence traced to parallel editing: local terminal work + direct edits on GitHub's web UI simultaneously. Resolved by standardizing on local-edit-then-push only (9/15)

---

## `archive/8-30-2026_documentation.md` — Why Archived, and How to Still View It

**[View the original file directly on GitHub →](https://github.com/william-gan-tech/biomechanics-project/blob/main/archive/8-30-2026_documentation.md)**

This file was an earlier, more detailed draft covering the same Mode 1–5
dashboard walkthrough later documented in `DEMONSTRATION.md`. It contains
the same fabricated/placeholder metrics found in `DEMONSTRATION.md`'s
Modes 1–4 (e.g. "91.4% Cross-Subject Accuracy," hardcoded per-skater
constants) — confirmed via direct string search on 9/15.

Since `DEMONSTRATION.md` already covers the same material with a
correction banner applied, this file was redundant and **moved to
`archive/` rather than deleted**, so the original pre-audit documentation
remains available as a historical record of what the dashboard's
documentation claimed before the audit. Useful for anyone auditing the
audit itself — **not to be cited as current or accurate.**

For the code-level equivalent of this same before/after picture, see the
`v1-real-cross-skater-comparison` git tag (below), which marks the last
commit before this class of fabricated metric was corrected in the actual
running dashboard.

---

## Git Tags — Preserved Historical Milestones

Rather than duplicating old code as archived files, key functional
milestones are preserved as permanent, browsable git tags:

| Tag | Marks | Link |
|---|---|---|
| `v1-real-cross-skater-comparison` | Real DTW-based cross-skater comparison replaces `np.random` fake data (9/09) | [View →](https://github.com/william-gan-tech/biomechanics-project/tree/v1-real-cross-skater-comparison) |
| `v2-repo-cleanup-complete` | Repository cleanup: archived legacy scripts, untracked ephemeral files, removed nested duplicate folder (9/12) | [View →](https://github.com/william-gan-tech/biomechanics-project/tree/v2-repo-cleanup-complete) |
| `v3-multiperson-tracking-fix` | Multi-person identity tracking + jitter smoothing fix, tracking diagnostic tool (9/15) | [View →](https://github.com/william-gan-tech/biomechanics-project/tree/v3-multiperson-tracking-fix) |

---

## Documentation-by-Phase Index

A running index of every documentation file, its phase, and its
verification status, so nothing has to be hunted down individually.
Expand this section as new docs and phases are added.

### Phase 1
- `docs/CAPABILITIES_PHASE1.md` — status: not independently re-verified during the 9/12 audit; not contradicted by anything found

### Phase 2
- `docs/CAPABILITIES_PHASE2.md` — status: mostly verified; ONNX claim downgraded (see table above)

### Phase 3
- `docs/CAPABILITIES_PHASE3.md` — status: fully verified, includes retractions and the outlier-sensitivity finding
- `archive/8-30-2026_documentation.md` — status: **retired, pre-audit, contains fabricated metrics** (see above)
- `docs/DEMONSTRATION.md` — status: corrected with banner; Modes 2–4 screenshots still need retaking from the fixed dashboard

### Phase 4 (once started)
- *(add entries here as Phase 4 documentation is created)*

---

## Workflow Note (for future reference)

This audit surfaced a recurring cause of confusion: documentation edited
in two places at once (locally + directly on GitHub's web UI) causes git
history to diverge repeatedly. Going forward: **edit files locally
(Notepad), then `git add` / `commit` / `push` — never paste directly into
GitHub's web editor.**