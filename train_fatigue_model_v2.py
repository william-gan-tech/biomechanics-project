"""
Retrains the fatigue autoencoder (v2), replacing the collapsed
skating_degradation_model.pth.

WHY (found 9/28 during Phase 5f): the saved model returns the SAME output
(knees ~134-135 deg, fixed positions) for zeros, random noise and noise x10
-- it collapsed to predicting one average pose. Root cause:
archive/phase1_2_scripts/train.py trained on RAW features (normalize=False),
so ~135-degree knee angles dominated the MSE and the LSTM learned the mean;
pipeline_engine.py then feeds it STANDARDIZED inputs on a different scale.

What v2 changes:
  - Angle-only inputs (knees, torso lean both sides). Position features were
    shown 9/28 to shift with broadcast camera zoom.
  - Each skater standardized by their OWN fresh-portion stats -- the same
    scheme validate_fatigue_form_degradation.py uses at scoring time (Phase 3
    rule: the late/fatigued data never sets its own baseline).
  - Windows only over consecutive frames (no windows spanning filtered gaps).
  - Held-out skater evaluation + explicit collapse checks, saved with the
    model: must respond to its input and beat trivial baselines, or the
    script says so loudly.
  - Saved as skating_fatigue_model_v2.pth with its feature list and
    standardization scheme. The old model and app.py are NOT changed.

Training data: the first half ("fresh" proxy, as in Phase 3) of every cached
skater video EXCEPT held-out test subjects (Sander Eitrem = Phase 5f test
subject). This learns "normal skating motion", as in Phase 1.

Usage:
    python -m train_fatigue_model_v2
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from model import SkatingLSTMAutoencoder  # noqa: E402

FEATURES = ["left_knee_filtered", "right_knee_filtered",
            "torso_lean_angle_deg", "torso_lean_angle_deg_left"]
TRAIN_VIDEOS = {
    "Sven Kramer": "data/sven_kramer_ref.mp4",
    "Patrick Meek": "data/patrick_meek_3000m.mp4",
    "Haralds Silovs": "data/silovs.mp4",
    "Ragne Wiklund": "data/ragne_wiklund.mp4",
    "start_candidate_2": "data/start_candidate_2.mp4",
    "start_candidate_3": "data/start_candidate_3.mp4",
}
EXCLUDED_TEST_SUBJECTS = ["Sander Eitrem"]  # never trained on (Phase 5f test subject)
VALIDATION_SKATER = "Patrick Meek"          # held out for the collapse/quality checks
FRESH_FRACTION = 0.5
WINDOW = 30
STRIDE = 2                # window step (9/28: 5 gave too few windows -- 1044 -- to model within-window motion)
MAX_WINDOWS_PER_SKATER = 1500
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
SEED = 0
OUT_PATH = "skating_fatigue_model_v2.pth"


def consecutive_windows(df):
    """Windows of WINDOW consecutive frames only, standardized by this
    skater's own stats (computed on the same fresh portion)."""
    df = df.dropna(subset=FEATURES).sort_values("frame")
    mean, std = df[FEATURES].mean().values, df[FEATURES].std().values + 1e-8
    arr = ((df[FEATURES].values - mean) / std).astype(np.float32)
    frames = df["frame"].values
    starts = [i for i in range(0, len(arr) - WINDOW + 1, STRIDE)
              if frames[i + WINDOW - 1] - frames[i] == WINDOW - 1]
    if len(starts) > MAX_WINDOWS_PER_SKATER:
        starts = np.linspace(0, len(starts) - 1, MAX_WINDOWS_PER_SKATER).astype(int).tolist()
        starts = [s for s in starts]
    return np.stack([arr[s:s + WINDOW] for s in starts]) if starts else np.zeros((0, WINDOW, len(FEATURES)), np.float32)


def load_fresh(skater, video):
    df = get_skater_features(skater, video, "scaled")
    if df is None:
        return None
    df = add_start_phase_features(filter_implausible_frames(df))
    df = df.sort_values("frame")
    return df.iloc[: int(len(df) * FRESH_FRACTION)]


def per_window_mse(model, x):
    model.eval()
    with torch.no_grad():
        recon, _ = model(torch.tensor(x))
    return ((recon.numpy() - x) ** 2).mean(axis=(1, 2)), recon.numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    windows = {}
    for skater, video in TRAIN_VIDEOS.items():
        if skater in EXCLUDED_TEST_SUBJECTS:
            continue
        df = load_fresh(skater, video)
        if df is None:
            print(f"  [SKIP] {skater}: no features")
            continue
        w = consecutive_windows(df)
        print(f"  {skater:20s}: {len(w):5d} windows from fresh {FRESH_FRACTION:.0%}")
        if len(w):
            windows[skater] = w

    val = windows.pop(VALIDATION_SKATER)
    train = np.concatenate(list(windows.values()))
    print(f"\nTraining on {len(train)} windows from {len(windows)} skaters; "
          f"validating on {VALIDATION_SKATER} ({len(val)} windows, never trained on)")

    model = SkatingLSTMAutoencoder(seq_len=WINDOW, n_features=len(FEATURES), embedding_dim=64, num_phases=3)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    t = torch.tensor(train)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(t))
        total, nb = 0.0, 0
        for i in range(0, len(t), BATCH_SIZE):
            batch = t[perm[i:i + BATCH_SIZE]]
            opt.zero_grad()
            recon, _ = model(batch)
            loss = crit(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            vloss, _ = per_window_mse(model, val)
            print(f"  epoch {epoch + 1:2d}/{EPOCHS}  train={total / nb:.4f}  held-out={vloss.mean():.4f}")

    # ---------------- collapse / quality checks ----------------
    vloss, recon = per_window_mse(model, val)
    baseline_global = float((val ** 2).mean())  # predict 0 = the skater's own mean
    baseline_window_mean = float(((val - val.mean(axis=1, keepdims=True)) ** 2).mean())  # flat line at each window's mean
    responsiveness = float(recon.std(axis=0).mean() / (val.std(axis=0).mean() + 1e-8))
    with torch.no_grad():
        z, _ = model(torch.zeros(1, WINDOW, len(FEATURES)))
        n, _ = model(torch.randn(1, WINDOW, len(FEATURES)))
    zero_vs_noise = float((z - n).abs().max())

    checks = {
        "held_out_mse": float(vloss.mean()),
        "baseline_predict_mean_mse": baseline_global,
        "baseline_flat_window_mse": baseline_window_mean,
        "responsiveness_output_vs_input_spread": responsiveness,
        "max_output_diff_zeros_vs_noise": zero_vs_noise,
    }
    checks["passes"] = bool(checks["held_out_mse"] < 0.8 * baseline_global
                            and checks["held_out_mse"] < baseline_window_mean
                            and responsiveness > 0.3 and zero_vs_noise > 0.05)

    print(f"\n{'='*70}\nCOLLAPSE / QUALITY CHECKS (held-out skater: {VALIDATION_SKATER})\n{'='*70}")
    print(f"  held-out reconstruction MSE           : {checks['held_out_mse']:.4f}")
    print(f"  baseline: always predict skater mean  : {baseline_global:.4f}  (the old model's failure mode)")
    print(f"  baseline: flat line at window mean    : {baseline_window_mean:.4f}  (must beat: shows it models motion)")
    print(f"  responsiveness (output/input spread)  : {responsiveness:.2f}  (collapsed model ~ 0)")
    print(f"  output change, zeros vs noise input   : {zero_vs_noise:.3f}  (old model: 0.0001)")
    print(f"  PASSES: {checks['passes']}")

    torch.save({
        "state_dict": model.state_dict(),
        "feature_cols": FEATURES,
        "window": WINDOW,
        "standardization": "per-skater, using that skater's own fresh/early-race stats",
        "trained_on": sorted(windows),
        "validation_skater": VALIDATION_SKATER,
        "excluded_test_subjects": EXCLUDED_TEST_SUBJECTS,
        "checks": checks,
    }, OUT_PATH)
    with open(OUT_PATH.replace(".pth", "_checks.json"), "w") as f:
        json.dump(checks, f, indent=2)
    print(f"\nSaved -> {OUT_PATH} (old skating_degradation_model.pth and app.py unchanged)")
    if not checks["passes"]:
        print("WARNING: model did NOT pass the quality checks -- do not use it for 5f conclusions.")


if __name__ == "__main__":
    main()
