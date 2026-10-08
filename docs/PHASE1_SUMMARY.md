# Phase 1 Summary & Conclusion

> **Correction (added 10/7):** On 9/28 the saved Phase 1 model
> (`skating_degradation_model.pth`) was found to be **collapsed**: it
> returns the same output (knees ~134-135°, fixed positions) for zeros,
> random noise or noise x10. It was trained on raw, unstandardized
> features (`archive/phase1_2_scripts/train.py`, `normalize=False`), so
> ~135° knee angles dominated the loss and the network learned one average
> pose. The model file used by the app and the ONNX export therefore does
> not detect fatigue. A retrained version with collapse checks
> (`skating_fatigue_model_v2.pth`) showed no reliable late-race signal
> across 9 skaters. The original Phase 1 experiment's write-up below is
> kept as a historical record. See `docs/CAPABILITIES_PHASE5.md` (5f).

## The Question
To what extent can deep learning architectures utilize comparative
temporal joint-angle trajectories across discrete video segments to
proactively forecast biomechanical performance degradation prior to
observable athletic deceleration in elite speed skaters?

## What Was Actually Tested
A single subject's long-form time trial footage (`skater_time_trial.mp4`)
was manually segmented into a fresh-state window (frames 500–1250) and a
fatigued-state window (frames 5625–6350), based on position within the
session. MediaPipe extracted 3D joint coordinates, a Butterworth low-pass
filter removed camera jitter, and an unsupervised PyTorch autoencoder was
trained exclusively on the fresh-state window to learn a baseline of
efficient movement. Reconstruction error (MSE) on held-out windows served
as the anomaly signal. Multiple architectures (feed-forward autoencoder,
LSTM autoencoder, TCN) were compared for their ability to capture temporal
dependencies.

## The Result
The fatigued-state window produced measurably higher reconstruction error
than the fresh-state window, and a predictive lead-time experiment found
that error spikes preceded measurable velocity decay in the underlying
hip-landmark trajectory — i.e., the model flagged mechanical drift before
the skater visibly slowed down.

## Why This Result Should Be Read With Appropriate Confidence
This phase's claims were **not independently re-run or re-verified**
during the 9/12–9/15 audit — the audit focused on Phase 2 and Phase 3
claims specifically, since those were the ones later found to contain
fabricated numbers. Nothing found during the audit contradicts Phase 1's
claims, but "not contradicted" is a different, weaker standard than
"independently re-confirmed." Phase 1's original code
(`demo_skater_a.py`, `train.py`, `evaluate.py`, and related scripts) is
preserved in `archive/phase1_2_scripts/` and could be re-run to close
this gap if a higher confidence level is needed later.

## Honest Limitations
- **n = 1 skater.** This is a single-subject proof of concept, not a
  cross-subject claim — that question wasn't asked until Phase 3.
- **Fresh/fatigued boundaries were manually chosen**, not derived from any
  physiological measurement — an assumption about where fatigue onset
  occurs in the session, not a verified ground truth.
- **No statistical significance testing** was applied to the fresh-vs-
  fatigued reconstruction error gap in the original Phase 1 work — this is
  a real gap relative to the rigor later applied in Phase 3, worth noting
  rather than glossing over.

## What It Actually Means
Phase 1 established that the core technical approach — autoencoder
reconstruction error as a fatigue-anomaly signal — is directionally
sound for at least one subject, and that the signal can precede visible
deceleration. It did not establish whether this generalizes across
subjects (that became Phase 3's question) or whether it holds up under
formal statistical testing (a gap Phase 3 addressed methodologically that
Phase 1 did not).

## What This Set Up for Later Phases
Phase 1's fresh/fatigued proxy-labeling technique (position-in-session as
a stand-in for verified fatigue state) was reused directly in Phase 3b's
fatigue-separability ablation — same assumption, same honest caveat,
applied across 7 subjects instead of 1.