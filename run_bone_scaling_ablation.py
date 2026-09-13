"""
Phase 3 ablation experiment: does bone-length scaling improve cross-subject
generalization of the fatigue-detection autoencoder?

Design (why it's set up this way):
  - We use Leave-One-Skater-Out (LOSO) cross-validation: train the
    autoencoder on N-1 skaters, evaluate reconstruction loss on the held-out
    skater, repeat for each skater. This directly measures "does this model
    generalize to an athlete it never saw."
  - We run this TWICE per fold: once with reference_scale=None (the old
    per-frame torso normalization) and once with a fixed calibrated
    bone-length scale. Same videos, same architecture, same training
    procedure -- the ONLY thing that changes is the normalization.
  - CRITICAL: standardization (mean/std) is computed ONCE from the pooled
    TRAINING skaters only, then applied to the held-out skater's data
    unchanged. If we z-scored each skater independently, we'd erase the very
    thing bone-scaling is supposed to fix (camera-distance/zoom differences
    between skaters) before the model ever saw it, making the ablation
    meaningless. Standardizing from training stats only and applying that
    to a held-out skater is what actually tests "does this held-out
    skater's data fit the distribution learned from other skaters."
  - The headline metric is the VARIANCE (or std) of held-out reconstruction
    loss ACROSS skaters, not just the mean. Lower cross-subject variance =
    more consistent generalization to new athletes. Mean loss level matters
    too, but variance is the direct "generalization" metric your research
    question asks about.

Usage:
    python -m run_bone_scaling_ablation

Outputs:
    ablation_results/results.csv       -- per-skater, per-condition loss
    ablation_results/summary.csv       -- headline variance/mean comparison
    ablation_results/comparison.png    -- bar chart of held-out loss
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pipeline_engine import compute_video_reference_scale
from preprocess_video import process_skating_video_multivariate
from model import SkatingLSTMAutoencoder

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, "ablation_feature_cache")
RESULTS_DIR = os.path.join(ROOT_DIR, "ablation_results")

# Skaters with real local video. Add more here as you get footage --
# more skaters = a more meaningful cross-subject variance estimate.
SKATER_VIDEOS = {
    "Sven Kramer": "data/sven_kramer_ref.mp4",
    "Patrick Meek": "data/patrick_meek_3000m.mp4",
    "Haralds Silovs": "data/silovs.mp4",
    "Ragne Wiklund": "data/ragne_wiklund.mp4",
    "Mia Manganello Kilburg": "data/mia_manganello_kilburg.mp4",
    "Jorrit Bergsma": "data/jorrit_bergsma.mp4",
    "Jan Blokhuijsen": "data/jan_blokhuijsen.mp4",
}

FEATURE_COLS = [
    'left_knee_filtered', 'right_knee_filtered',
    'norm_right_hip_x', 'norm_right_hip_y',
    'norm_right_shoulder_x', 'norm_right_shoulder_y'
]

WINDOW_SIZE = 30
EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_WINDOWS_PER_SKATER = 1500  # caps training-pool size for speed; applied
                                # identically to both conditions so it doesn't
                                # bias the comparison, just speeds it up


# ============================================================
# Feature extraction + caching (per skater, per condition)
# ============================================================

def _cache_path(skater_name, condition):
    safe = "".join(c if c.isalnum() else "_" for c in skater_name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    return (
        os.path.join(CACHE_DIR, f"{safe}_{condition}_features.csv"),
        os.path.join(CACHE_DIR, f"{safe}_{condition}_meta.json"),
    )


def get_skater_features(skater_name, video_rel_path, condition):
    """condition is 'unscaled' or 'scaled'. Returns a DataFrame of raw
    (NOT standardized) features, or None if extraction failed."""
    full_path = os.path.join(ROOT_DIR, video_rel_path)
    if not os.path.exists(full_path):
        print(f"  [SKIP] {skater_name}: video not found at {full_path}")
        return None

    cache_csv, cache_meta = _cache_path(skater_name, condition)
    video_mtime = os.path.getmtime(full_path)

    if os.path.exists(cache_csv) and os.path.exists(cache_meta):
        try:
            with open(cache_meta, "r") as f:
                meta = json.load(f)
            if meta.get("video_mtime") == video_mtime:
                return pd.read_csv(cache_csv)
        except Exception:
            pass

    cap = cv2.VideoCapture(full_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    reference_scale = compute_video_reference_scale(full_path) if condition == "scaled" else None

    df = process_skating_video_multivariate(full_path, fps=fps, reference_scale=reference_scale)
    if df is None or df.empty:
        print(f"  [FAIL] {skater_name}: feature extraction returned nothing")
        return None

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    try:
        df.to_csv(cache_csv, index=False)
        with open(cache_meta, "w") as f:
            json.dump({"video_mtime": video_mtime, "reference_scale": reference_scale}, f)
    except Exception:
        pass

    return df


def filter_implausible_frames(df, max_abs_position=3.0):
    """Drops frames where bone-scaled hip/shoulder position values are
    implausibly large -- a strong signal the full body wasn't actually in
    frame (e.g. a broadcast cutaway to a close-up shot of just the skates/
    knees), rather than a real, trackable pose. Properly bone-scaled
    positions relative to the hip midpoint should rarely exceed +/-2 to 3
    units when the whole body is visible.

    Applied identically to every skater and both conditions (not just to
    "fix" one inconvenient result) so it's a principled data-quality rule,
    not selective post-hoc editing.
    """
    position_cols = ['norm_right_hip_x', 'norm_right_hip_y', 'norm_right_shoulder_x', 'norm_right_shoulder_y']
    mask = pd.Series(True, index=df.index)
    for col in position_cols:
        if col in df.columns:
            mask &= df[col].abs() <= max_abs_position

    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  [FILTER] Dropped {n_dropped}/{len(df)} frames with implausible position values "
              f"(likely camera cutaways / partial-body framing)")

    return df[mask].reset_index(drop=True)


def make_windows(df, window_size=WINDOW_SIZE, max_windows=MAX_WINDOWS_PER_SKATER):
    """Returns an array of shape (n_windows, window_size, n_features).
    If more than `max_windows` are available, evenly subsamples down to
    that count (rather than truncating to just the start of the clip) to
    keep training time bounded without biasing toward one part of the
    session."""
    arr = df[FEATURE_COLS].values.astype(np.float32)
    if len(arr) < window_size:
        return np.zeros((0, window_size, len(FEATURE_COLS)), dtype=np.float32)
    windows = [arr[i:i + window_size] for i in range(len(arr) - window_size + 1)]
    windows = np.array(windows, dtype=np.float32)

    if max_windows is not None and len(windows) > max_windows:
        idx = np.linspace(0, len(windows) - 1, max_windows).astype(int)
        windows = windows[idx]

    return windows


# ============================================================
# Training / evaluation
# ============================================================

def train_and_eval(train_windows, test_windows, epochs=EPOCHS):
    """Standardizes using TRAINING data stats only, trains a fresh
    autoencoder on the training skaters' windows, and returns the mean
    reconstruction loss on the held-out (test) skater's windows.
    """
    # Standardize using TRAINING pooled stats only (see module docstring for why)
    flat_train = train_windows.reshape(-1, train_windows.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-8

    train_norm = (train_windows - mean) / std
    test_norm = (test_windows - mean) / std

    model = SkatingLSTMAutoencoder(
        seq_len=WINDOW_SIZE, n_features=len(FEATURE_COLS), embedding_dim=64, num_phases=3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(train_norm, dtype=torch.float32)
    n = len(train_tensor)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch = train_tensor[idx]
            optimizer.zero_grad()
            reconstruction, _phase_logits = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"      epoch {epoch+1}/{epochs}  train_loss={epoch_loss/max(n_batches,1):.5f}")

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_norm, dtype=torch.float32)
        reconstruction, _ = model(test_tensor)
        per_window_loss = torch.mean((reconstruction - test_tensor) ** 2, dim=(1, 2))
        mean_test_loss = per_window_loss.mean().item()

    return mean_test_loss


def run_loso_for_condition(skater_windows, condition_name, results_csv_path):
    """skater_windows: dict[skater_name -> np.ndarray windows]. Runs LOSO
    across all skaters that have at least a few windows.

    Saves results incrementally to `results_csv_path` after EVERY fold (not
    just at the end), and skips any (skater, condition) pair already present
    in that file -- so killing this script partway through and re-running it
    picks up where it left off instead of starting over.
    """
    usable = {k: v for k, v in skater_windows.items() if len(v) >= 5}
    skaters = list(usable.keys())
    results = []

    # Load any already-completed results so we can skip them
    completed = set()
    if os.path.exists(results_csv_path):
        try:
            existing = pd.read_csv(results_csv_path)
            completed = set(zip(existing["skater"], existing["condition"]))
            results = existing.to_dict("records")
        except Exception:
            pass

    for held_out in skaters:
        if (held_out, condition_name) in completed:
            print(f"    Held-out: {held_out}  [SKIPPED -- already completed, found in {results_csv_path}]")
            continue

        print(f"    Held-out: {held_out}")
        train_list = [usable[s] for s in skaters if s != held_out]
        train_windows = np.concatenate(train_list, axis=0)
        test_windows = usable[held_out]

        mean_loss = train_and_eval(train_windows, test_windows)
        print(f"      -> held-out reconstruction loss: {mean_loss:.5f}")
        results.append({"skater": held_out, "condition": condition_name, "held_out_loss": mean_loss})

        # Save after EVERY fold -- this is what makes the script resumable
        pd.DataFrame(results).to_csv(results_csv_path, index=False)
        print(f"      (saved progress -> {results_csv_path})")

    return [r for r in results if r["condition"] == condition_name]


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Extracting features (unscaled + scaled) per skater")
    print("=" * 60)

    windows_by_condition = {"unscaled": {}, "scaled": {}}

    for skater_name, rel_path in SKATER_VIDEOS.items():
        for condition in ["unscaled", "scaled"]:
            print(f"\n{skater_name} [{condition}]")
            df = get_skater_features(skater_name, rel_path, condition)
            if df is None:
                continue
            df = filter_implausible_frames(df)
            windows = make_windows(df)
            print(f"  -> {len(windows)} windows of shape {windows.shape[1:]}")
            windows_by_condition[condition][skater_name] = windows

    print("\n" + "=" * 60)
    print("STEP 2: Leave-One-Skater-Out training + evaluation")
    print("=" * 60)

    results_csv_path = os.path.join(RESULTS_DIR, "results.csv")

    all_results = []
    for condition in ["unscaled", "scaled"]:
        print(f"\n--- Condition: {condition} ---")
        results = run_loso_for_condition(windows_by_condition[condition], condition, results_csv_path)
        all_results.extend(results)

    results_df = pd.read_csv(results_csv_path)  # reload full accumulated results (handles resumed runs)

    print("\n" + "=" * 60)
    print("STEP 3: Summary -- does bone-length scaling reduce cross-subject variance?")
    print("=" * 60)

    summary_rows = []
    for condition in ["unscaled", "scaled"]:
        subset = results_df[results_df["condition"] == condition]["held_out_loss"]
        if len(subset) == 0:
            continue
        summary_rows.append({
            "condition": condition,
            "n_skaters": len(subset),
            "mean_held_out_loss": subset.mean(),
            "std_held_out_loss": subset.std(),
            "variance_held_out_loss": subset.var(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(summary_df.to_string(index=False))

    if len(summary_df) == 2:
        unscaled_var = summary_df[summary_df["condition"] == "unscaled"]["variance_held_out_loss"].values[0]
        scaled_var = summary_df[summary_df["condition"] == "scaled"]["variance_held_out_loss"].values[0]
        if unscaled_var > 0:
            pct_change = (scaled_var - unscaled_var) / unscaled_var * 100
            direction = "REDUCED" if pct_change < 0 else "INCREASED"
            print(f"\nBone-length scaling {direction} cross-subject loss variance by {abs(pct_change):.1f}%")
            print("(Negative % change = scaling made held-out reconstruction loss more consistent")
            print(" across different skaters -- supporting improved cross-subject generalization.)")

    # Plot: per-skater held-out loss, unscaled vs scaled, side by side
    pivot = results_df.pivot(index="skater", columns="condition", values="held_out_loss")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Held-out Reconstruction Loss (MSE)")
        ax.set_title("LOSO Held-Out Reconstruction Loss: Unscaled vs. Bone-Scaled")
        ax.legend(title="Condition")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "comparison.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nSaved plot -> {plot_path}")

    print(f"\nFull results -> {results_csv_path}")
    print(f"Summary -> {summary_csv_path}")


if __name__ == "__main__":
    main()
