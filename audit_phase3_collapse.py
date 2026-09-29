"""
Phase 5m audit (9/29): were the Phase 3b fatigue-separability models
collapsed like the original skating_degradation_model.pth?

On 9/28 the saved Phase 1 model was found to return the same output for any
input (trained on unstandardized features). Phase 3b
(run_fatigue_separability_ablation.py) trains its OWN models, with
standardization, so it is less likely to share the problem -- but it trains
for only 8 epochs on position-based features, and its conclusions rest on
reconstruction-loss gaps, which a collapsed model would produce meaninglessly.

This re-runs Phase 3b's exact training (same function, same data, same
split) for a sample of leave-one-skater-out folds, keeps each trained model,
and applies the same collapse checks as train_fatigue_model_v2.py:
  - held-out fresh MSE vs "always predict the mean" (collapsed-model failure)
  - held-out fresh MSE vs "flat line at each window's mean" (must beat it to
    be modelling motion rather than just level)
  - responsiveness: spread of outputs relative to spread of inputs (~0 = collapsed)
  - output change between zeros and random-noise input

Does NOT modify Phase 3 results. Training is random, so losses won't match
the original run exactly; the question is only whether models of this
recipe collapse.

Usage:
    python -m audit_phase3_collapse
"""

import numpy as np
import pandas as pd
import torch

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames, SKATER_VIDEOS, FEATURE_COLS, WINDOW_SIZE
from run_fatigue_separability_ablation import split_fresh_fatigued, make_windows, train_on_fresh_eval_both

CONDITIONS = ["unscaled", "scaled", "zscore_only"]
FOLDS_PER_CONDITION = 3     # a sample of held-out skaters, to keep runtime bounded
MAX_WINDOWS = 3000          # per skater per segment, evenly subsampled (speed only)
OUTPUT_PATH = "phase3_collapse_audit.csv"
SEED = 0


def subsample(w):
    if len(w) <= MAX_WINDOWS:
        return w
    return w[np.linspace(0, len(w) - 1, MAX_WINDOWS).astype(int)]


def checks(model, x):
    model.eval()
    with torch.no_grad():
        recon, _ = model(torch.tensor(x))
        recon = recon.numpy()
        z, _ = model(torch.zeros(1, WINDOW_SIZE, len(FEATURE_COLS)))
        n, _ = model(torch.randn(1, WINDOW_SIZE, len(FEATURE_COLS)))
    mse = float(((recon - x) ** 2).mean())
    predict_mean = float((x ** 2).mean())
    flat_window = float(((x - x.mean(axis=1, keepdims=True)) ** 2).mean())
    resp = float(recon.std(axis=0).mean() / (x.std(axis=0).mean() + 1e-8))
    zvn = float((z - n).abs().max())
    return {
        "held_out_fresh_mse": mse, "baseline_predict_mean": predict_mean,
        "baseline_flat_window": flat_window, "responsiveness": resp, "zeros_vs_noise": zvn,
        "collapsed": bool(resp < 0.1 or zvn < 0.01 or mse >= 0.95 * predict_mean),
        "models_motion": bool(mse < flat_window),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rows = []
    for condition in CONDITIONS:
        data = {}
        for skater, path in SKATER_VIDEOS.items():
            df = get_skater_features(skater, path, condition)
            if df is None:
                continue
            fresh, fatigued = split_fresh_fatigued(filter_implausible_frames(df))
            fw, gw = make_windows(fresh), make_windows(fatigued)
            if len(fw) >= 5 and len(gw) >= 5:
                data[skater] = (subsample(fw), subsample(gw))
        skaters = list(data)
        print(f"\n--- {condition}: {len(skaters)} usable skaters; auditing {min(FOLDS_PER_CONDITION, len(skaters))} folds ---")
        for held_out in skaters[:FOLDS_PER_CONDITION]:
            train = np.concatenate([data[s][0] for s in skaters if s != held_out])
            fresh_loss, fatigued_loss, model, test_fresh_norm = train_on_fresh_eval_both(
                train, data[held_out][0], data[held_out][1], return_model=True)
            c = checks(model, test_fresh_norm)
            c.update({"condition": condition, "held_out": held_out,
                      "fresh_loss": fresh_loss, "fatigued_loss": fatigued_loss})
            rows.append(c)
            print(f"    {held_out:24s} mse={c['held_out_fresh_mse']:.3f}  predict-mean={c['baseline_predict_mean']:.3f}  "
                  f"flat={c['baseline_flat_window']:.3f}  resp={c['responsiveness']:.2f}  "
                  f"collapsed={c['collapsed']}  models_motion={c['models_motion']}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(out.groupby("condition")[["collapsed", "models_motion"]].mean().rename(
        columns={"collapsed": "frac_collapsed", "models_motion": "frac_models_motion"}).round(2).to_string())
    print("\nREADING THIS:")
    print("  - frac_collapsed > 0: some Phase 3b models ignore their input -> their loss gaps are not meaningful.")
    print("  - frac_models_motion < 1: models only capture each window's level, not within-window motion;")
    print("    loss gaps then mostly reflect posture/level shifts, not movement quality.")
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
