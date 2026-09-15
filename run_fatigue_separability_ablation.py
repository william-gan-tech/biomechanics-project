"""
Phase 3b: Does bone-length scaling improve cross-subject FATIGUE DETECTION
(not just general motion-reconstruction generalization)?

This is a different, more targeted experiment than run_bone_scaling_ablation.py.
That script (Phase 3a) tested whether scaling reduces cross-subject variance in
GENERAL reconstruction loss -- it never distinguished fresh from fatigued
motion at all. This script specifically tests fatigue detection:

  1. For each skater, split their session into an early ("fresh" proxy) and
     late ("fatigued" proxy) segment -- the same early-vs-late-session
     technique used in Phase 1 for skater_time_trial.mp4. This is an
     ASSUMPTION (endurance events accumulate fatigue over the session), not
     verified ground-truth fatigue labeling. Documented explicitly rather
     than silently assumed.
  2. Leave-One-Skater-Out: train an autoencoder ONLY on the FRESH segments
     of the other N-1 skaters (matching the original unsupervised-autoencoder
     philosophy from Phase 1 -- learn what "normal" motion looks like).
  3. Evaluate the trained model on the held-out skater's FRESH segment AND
     their FATIGUED segment, using standardization stats derived only from
     training-skaters' fresh data (never touching the held-out skater's
     stats, for the same reason as Phase 3a -- don't erase the effect being
     tested before the model sees it).
  4. The key metric is SEPARABILITY: how much higher is reconstruction loss
     on the held-out skater's fatigued segment vs their fresh segment. A
     model that generalizes fatigue detection to a NEW, unseen athlete
     should show a large, CONSISTENT (low cross-subject variance)
     separability gap. This directly tests the original Phase 3 research
     question, unlike the Phase 3a variance-of-general-loss metric.

Usage:
    python -m run_fatigue_separability_ablation

Outputs:
    fatigue_separability_results/results.csv    -- per-skater, per-condition fresh/fatigued loss + gap
    fatigue_separability_results/summary.csv    -- headline separability comparison
    fatigue_separability_results/comparison.png -- bar chart
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from run_bone_scaling_ablation import (
    get_skater_features,
    filter_implausible_frames,
    SKATER_VIDEOS,
    FEATURE_COLS,
    WINDOW_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
)
from model import SkatingLSTMAutoencoder

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "fatigue_separability_results")

EPOCHS = 8
FRESH_FRACTION = 0.25   # first 25% of session = "fresh" proxy
FATIGUED_FRACTION = 0.25  # last 25% of session = "fatigued" proxy
# Middle 50% is deliberately dropped -- keeps fresh/fatigued segments
# clearly separated in time rather than adjacent, same logic as Phase 1's
# non-contiguous frame-range approach.


def split_fresh_fatigued(df, fresh_frac=FRESH_FRACTION, fatigued_frac=FATIGUED_FRACTION):
    """Splits a skater's feature DataFrame (must have a 'frame' column, in
    order) into early-session ("fresh" proxy) and late-session ("fatigued"
    proxy) segments, dropping the middle."""
    df = df.sort_values("frame").reset_index(drop=True)
    n = len(df)
    fresh_end = int(n * fresh_frac)
    fatigued_start = int(n * (1 - fatigued_frac))
    fresh_df = df.iloc[:fresh_end].reset_index(drop=True)
    fatigued_df = df.iloc[fatigued_start:].reset_index(drop=True)
    return fresh_df, fatigued_df


def make_windows(df, window_size=WINDOW_SIZE):
    if df is None or len(df) < window_size:
        return np.zeros((0, window_size, len(FEATURE_COLS)), dtype=np.float32)
    arr = df[FEATURE_COLS].values.astype(np.float32)
    windows = [arr[i:i + window_size] for i in range(len(arr) - window_size + 1)]
    return np.array(windows, dtype=np.float32)


def train_on_fresh_eval_both(train_fresh_windows, test_fresh_windows, test_fatigued_windows, epochs=EPOCHS):
    """Trains ONLY on pooled fresh windows from training skaters (standardized
    using training-fresh stats only), then evaluates reconstruction loss on
    the held-out skater's fresh AND fatigued windows using those SAME stats.
    Returns (fresh_loss, fatigued_loss)."""
    flat_train = train_fresh_windows.reshape(-1, train_fresh_windows.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-8

    train_norm = (train_fresh_windows - mean) / std
    test_fresh_norm = (test_fresh_windows - mean) / std
    test_fatigued_norm = (test_fatigued_windows - mean) / std

    model = SkatingLSTMAutoencoder(seq_len=WINDOW_SIZE, n_features=len(FEATURE_COLS), embedding_dim=64, num_phases=3)
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
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 4 == 0 or epoch == epochs - 1:
            print(f"        epoch {epoch+1}/{epochs}  train_loss(fresh-only)={epoch_loss/max(n_batches,1):.5f}")

    model.eval()
    with torch.no_grad():
        fresh_tensor = torch.tensor(test_fresh_norm, dtype=torch.float32)
        fatigued_tensor = torch.tensor(test_fatigued_norm, dtype=torch.float32)

        fresh_recon, _ = model(fresh_tensor)
        fresh_loss = torch.mean((fresh_recon - fresh_tensor) ** 2, dim=(1, 2)).mean().item()

        fatigued_recon, _ = model(fatigued_tensor)
        fatigued_loss = torch.mean((fatigued_recon - fatigued_tensor) ** 2, dim=(1, 2)).mean().item()

    return fresh_loss, fatigued_loss


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_csv_path = os.path.join(RESULTS_DIR, "results.csv")

    print("=" * 70)
    print("STEP 1: Loading features + splitting into fresh/fatigued proxy segments")
    print("=" * 70)
    print(f"(Fresh = first {FRESH_FRACTION*100:.0f}% of session, "
          f"Fatigued = last {FATIGUED_FRACTION*100:.0f}%, middle dropped)")

    windows_by_condition = {"unscaled": {}, "scaled": {}, "zscore_only": {}}

    for skater_name, rel_path in SKATER_VIDEOS.items():
        for condition in ["unscaled", "scaled", "zscore_only"]:
            print(f"\n{skater_name} [{condition}]")
            df = get_skater_features(skater_name, rel_path, condition)
            if df is None:
                continue
            df = filter_implausible_frames(df)

            fresh_df, fatigued_df = split_fresh_fatigued(df)
            fresh_windows = make_windows(fresh_df)
            fatigued_windows = make_windows(fatigued_df)
            print(f"  -> fresh: {len(fresh_windows)} windows, fatigued: {len(fatigued_windows)} windows")

            if len(fresh_windows) >= 5 and len(fatigued_windows) >= 5:
                windows_by_condition[condition][skater_name] = {
                    "fresh": fresh_windows,
                    "fatigued": fatigued_windows,
                }
            else:
                print(f"  [SKIP] Not enough windows in fresh and/or fatigued segment for {skater_name}")

    print("\n" + "=" * 70)
    print("STEP 2: Leave-One-Skater-Out fatigue-separability evaluation")
    print("=" * 70)

    all_results = []
    completed = set()
    if os.path.exists(results_csv_path):
        try:
            existing = pd.read_csv(results_csv_path)
            completed = set(zip(existing["skater"], existing["condition"]))
            all_results = existing.to_dict("records")
        except Exception:
            pass

    for condition in ["unscaled", "scaled", "zscore_only"]:
        print(f"\n--- Condition: {condition} ---")
        usable = windows_by_condition[condition]
        skaters = list(usable.keys())

        for held_out in skaters:
            if (held_out, condition) in completed:
                print(f"    Held-out: {held_out}  [SKIPPED -- already completed]")
                continue

            print(f"    Held-out: {held_out}")
            train_fresh_list = [usable[s]["fresh"] for s in skaters if s != held_out]
            train_fresh_windows = np.concatenate(train_fresh_list, axis=0)

            test_fresh_windows = usable[held_out]["fresh"]
            test_fatigued_windows = usable[held_out]["fatigued"]

            fresh_loss, fatigued_loss = train_on_fresh_eval_both(
                train_fresh_windows, test_fresh_windows, test_fatigued_windows
            )
            gap = fatigued_loss - fresh_loss
            print(f"      -> fresh_loss={fresh_loss:.5f}  fatigued_loss={fatigued_loss:.5f}  "
                  f"SEPARABILITY GAP={gap:.5f}")

            all_results.append({
                "skater": held_out,
                "condition": condition,
                "fresh_loss": fresh_loss,
                "fatigued_loss": fatigued_loss,
                "separability_gap": gap,
            })
            pd.DataFrame(all_results).to_csv(results_csv_path, index=False)

    results_df = pd.read_csv(results_csv_path)

    print("\n" + "=" * 70)
    print("STEP 3: Summary -- does scaling improve cross-subject FATIGUE separability?")
    print("=" * 70)

    summary_rows = []
    for condition in ["unscaled", "scaled", "zscore_only"]:
        subset = results_df[results_df["condition"] == condition]
        if len(subset) == 0:
            continue
        summary_rows.append({
            "condition": condition,
            "n_skaters": len(subset),
            "mean_separability_gap": subset["separability_gap"].mean(),
            "std_separability_gap": subset["separability_gap"].std(),
            "variance_separability_gap": subset["separability_gap"].var(),
            "pct_skaters_with_positive_gap": (subset["separability_gap"] > 0).mean() * 100,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print(summary_df.to_string(index=False))

    print("\nInterpretation guide:")
    print("  - Higher mean_separability_gap = fatigued motion reconstructs worse than")
    print("    fresh motion on average -- the model IS detecting fatigue in unseen skaters.")
    print("  - Lower variance_separability_gap = that detection capability is MORE")
    print("    CONSISTENT across different unseen athletes -- this is the direct")
    print("    cross-subject generalization answer to the Phase 3 question.")
    print("  - pct_skaters_with_positive_gap = what fraction of held-out skaters showed")
    print("    ANY fatigue signal at all (gap > 0), regardless of magnitude.")

    pivot = results_df.pivot(index="skater", columns="condition", values="separability_gap")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_ylabel("Separability Gap (fatigued_loss - fresh_loss)")
        ax.set_title("Cross-Subject Fatigue-Detection Separability by Condition")
        ax.legend(title="Condition")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
        print(f"\nSaved plot -> {os.path.join(RESULTS_DIR, 'comparison.png')}")

    print(f"\nFull results -> {results_csv_path}")


if __name__ == "__main__":
    main()
