# Phase 2 Summary & Conclusion

## The Question
Can the Phase 1 proof-of-concept be turned into an automated,
end-to-end system — ingesting raw, unsegmented video (local upload or
YouTube URL) and producing fatigue telemetry without manual frame
selection?

## What Was Actually Built
`pipeline_engine.py` automates the full path from raw video to
reconstruction-loss timeline: `yt_dlp`-based ingestion, MediaPipe pose
extraction, Butterworth filtering, sliding-window autoencoder inference,
and automated statistical threshold calibration (mean + k·σ from initial
frames). This was wrapped in a Streamlit dashboard (`app.py`) with
persistent session state, dynamic threshold controls, annotated video
export, and CSV/report downloads.

## The Result — Verified Piece by Piece
Unlike Phase 1, most of Phase 2's claims **were** directly tested during
the 9/08–9/15 audit, not just reviewed on paper:

| Claim | Verification |
|---|---|
| End-to-end video ingestion pipeline | ✅ Confirmed functional — used repeatedly to process 7+ skaters' footage |
| YouTube URL ingestion via `yt_dlp` | ✅ Confirmed working 9/10, used to download 4 skaters' videos live |
| Streamlit dashboard, session state, threshold controls | ✅ Confirmed functional in `app.py` |
| Automated baseline calibration | ✅ Confirmed, though found to have real bugs (deprecated API, silent fallback failures) that were fixed 9/08–9/12 |
| ONNX edge runtime acceleration | ❌ **Found non-functional.** `onnxruntime` is imported but never called anywhere in the running code. The "int8" model file (626KB) is *larger* than the original FP32 file (133KB) with mixed tensor types — quantization was attempted but did not cleanly complete or get wired into any inference path. |
| Predictive lead-time analysis, temporal window refactoring | ⚠️ Not independently re-verified during this audit — same caveat as Phase 1 |

## Why the ONNX Finding Matters
This wasn't a documentation-only correction — it changed what the system
can actually be claimed to do. Earlier project materials described ONNX
as providing real CPU/GPU latency benefits for edge deployment; the
system currently runs standard PyTorch inference with no edge
acceleration path active. This has been corrected across all docs
(`README.md`, `CAPABILITIES_PHASE2.md`, `MILESTONES.md`) rather than left
standing.

## Honest Limitations
- Two claims (lead-time analysis, temporal window refactoring) remain in
  the "not independently re-verified" category, same as most of Phase 1.
- The gap between "ONNX files exist in the repo" and "ONNX inference
  actually runs" is exactly the kind of claim that's easy to state
  correctly-sounding without checking — worth remembering as a pattern
  for any future edge-deployment work in Phase 4+.

## What It Actually Means
Phase 2's core automation goal was achieved and independently confirmed —
the pipeline genuinely does take raw video to fatigue telemetry without
manual intervention. The one significant overstatement (ONNX) has been
corrected. This is the most thoroughly re-verified of the three phases,
since it's the layer Phase 3's real experiments were built directly on
top of and exercised repeatedly.