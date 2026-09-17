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
the 9/08–9/16 audit, not just reviewed on paper:

| Claim | Verification |
|---|---|
| End-to-end video ingestion pipeline | ✅ Confirmed functional — used repeatedly to process 7+ skaters' footage |
| YouTube URL ingestion via `yt_dlp` | ✅ Confirmed working 9/10, used to download 4 skaters' videos live |
| Streamlit dashboard, session state, threshold controls | ✅ Confirmed functional in `app.py` |
| Automated baseline calibration | ✅ Confirmed, though found to have real bugs (deprecated API, silent fallback failures) that were fixed 9/08–9/12 |
| ONNX edge runtime acceleration — FP32 | ✅ **Fixed and verified 9/16.** Rebuilt the export pipeline after finding the original was non-functional (see below). Numerically equivalent to PyTorch (max output difference: 0.000031) and **measured 3.08x faster** (0.918ms vs. 2.824ms mean, batch size 8, 200 runs, CPU) via a reproducible benchmark. A real `ONNXFatigueDetector` class now actually calls `onnxruntime.InferenceSession` — previously the import existed but was never invoked anywhere. |
| ONNX edge runtime acceleration — INT8 | ⚠️ **Partially fixed, still not usable.** The original "int8" file (626KB) was larger than the FP32 original (133KB) with mixed tensor types — quantization never cleanly completed. The file-size bug is now fixed (144KB vs. 495KB, a genuine 70.9% reduction), but output verification shows a max difference of 55.7 vs. PyTorch — the quantized model's outputs are functionally broken, not just less precise. Documented as a known limitation of dynamic INT8 quantization on this LSTM architecture, not claimed as working. |
| Predictive lead-time analysis, temporal window refactoring | ⚠️ Not independently re-verified during this audit — same caveat as Phase 1 |

## Why the ONNX Finding Matters
This wasn't a documentation-only correction — it changed what the system
can actually be claimed to do, twice over. Earlier project materials
described ONNX as providing real CPU/GPU latency benefits for edge
deployment while the system actually ran unaccelerated PyTorch inference
with a dead import. That was corrected 9/16 for the FP32 path specifically
— it now genuinely provides the claimed latency benefit, backed by a
real, reproducible benchmark rather than an assumption. The INT8 path,
however, is a case where **fixing one bug (file size) surfaced a second,
more serious bug (correctness)** — a reminder that verifying a claim
requires checking the actual output, not just that a process completes
without erroring.

## Honest Limitations
- Two claims (lead-time analysis, temporal window refactoring) remain in
  the "not independently re-verified" category, same as most of Phase 1.
- INT8 quantization remains genuinely unresolved — a working fix would
  likely require static (calibration-based) quantization or per-channel
  handling specific to recurrent/LSTM weights, neither of which has been
  attempted yet. This is flagged as real, open work, not swept under the
  "documented limitation" label as if it were finished.
- The gap between "ONNX files exist in the repo" and "ONNX inference
  actually runs and produces correct output" is exactly the kind of claim
  that's easy to state correctly-sounding without checking — the INT8
  case specifically shows that even "the file size looks right now" isn't
  sufficient verification on its own; output correctness has to be
  checked separately, every time.

## What It Actually Means
Phase 2's core automation goal was achieved and independently confirmed —
the pipeline genuinely does take raw video to fatigue telemetry without
manual intervention. The ONNX FP32 acceleration claim, previously
completely false, is now genuinely true and measured. The INT8
acceleration claim remains false, but is now honestly characterized as a
specific, diagnosed correctness bug rather than a vague "not integrated"
status. This is the most thoroughly re-verified of the three phases,
since it's the layer Phase 3's real experiments were built directly on
top of and exercised repeatedly — and, as of 9/16, the layer where a
previously-fabricated-adjacent claim was actually turned into a real,
working, benchmarked feature rather than just corrected on paper.