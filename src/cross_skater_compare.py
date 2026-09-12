"""
Real cross-skater comparison using bone-scaled joint features.

Replaces the previous np.random.uniform() demo data in Mode 1 with an
actual comparison of two skaters' joint trajectories:

  1. Each skater's video is calibrated (compute_video_reference_scale) and
     processed into per-frame features (process_skating_video_multivariate),
     with results cached to disk so repeat comparisons don't reprocess video.
  2. Each feature series is resampled to a fixed-length "normalized stride
     cycle" so skaters with different clip lengths/speeds are comparable.
  3. DTW (dynamic time warping) distance is computed per feature to measure
     how well-aligned the two skaters' motion patterns are, tolerating
     timing differences that plain frame-by-frame MSE would over-penalize.

Honesty note: this does NOT reproduce "Hip Angle" or "Ankle Dorsiflexion"
from the original placeholder UI -- those were never actually computed
anywhere in this pipeline. The real features available are: right/left
knee flexion angle, and bone-scaled hip/shoulder position (x, y). Labels
in the dashboard should reflect that.
"""

import os
import json
import numpy as np
import pandas as pd
import cv2

from pipeline_engine import compute_video_reference_scale
from preprocess_video import process_skating_video_multivariate

# Maps the actual columns produced by process_skating_video_multivariate to
# human-readable labels. Extend this if you add more features upstream.
FEATURE_COLUMNS = {
    "right_knee_filtered": "Right Knee Angle",
    "left_knee_filtered": "Left Knee Angle",
    "norm_right_hip_x": "Hip Position (X, bone-scaled)",
    "norm_right_hip_y": "Hip Position (Y, bone-scaled)",
    "norm_right_shoulder_x": "Shoulder Position (X, bone-scaled)",
    "norm_right_shoulder_y": "Shoulder Position (Y, bone-scaled)",
}


def _cache_paths(skater_name, root_dir):
    safe = "".join(c if c.isalnum() else "_" for c in skater_name)
    cache_dir = os.path.join(root_dir, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return (
        os.path.join(cache_dir, f"{safe}_features.csv"),
        os.path.join(cache_dir, f"{safe}_meta.json"),
    )


def get_or_compute_skater_features(skater_name, video_map, root_dir, force_recompute=False):
    """Returns (df_features, reference_scale) for a skater, using a disk
    cache keyed off the source video's modification time so re-running the
    dashboard doesn't reprocess video that hasn't changed.

    Returns (None, None) if no video path is configured for this skater in
    `video_map`, the file doesn't exist, or feature extraction fails.
    """
    rel_path = video_map.get(skater_name)
    if not rel_path:
        return None, None

    full_path = rel_path if os.path.isabs(rel_path) else os.path.join(root_dir, rel_path)
    if not os.path.exists(full_path):
        return None, None

    cache_csv, cache_meta = _cache_paths(skater_name, root_dir)
    video_mtime = os.path.getmtime(full_path)

    if not force_recompute and os.path.exists(cache_csv) and os.path.exists(cache_meta):
        try:
            with open(cache_meta, "r") as f:
                meta = json.load(f)
            if meta.get("video_mtime") == video_mtime:
                df = pd.read_csv(cache_csv)
                return df, meta.get("reference_scale", 0.45)
        except Exception:
            pass  # fall through and recompute

    try:
        reference_scale = compute_video_reference_scale(full_path)

        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        df = process_skating_video_multivariate(full_path, fps=fps, reference_scale=reference_scale)
        if df is None or df.empty:
            return None, None

        try:
            df.to_csv(cache_csv, index=False)
            with open(cache_meta, "w") as f:
                json.dump({"video_mtime": video_mtime, "reference_scale": reference_scale}, f)
        except Exception:
            pass  # caching is best-effort; still return the freshly computed result

        return df, reference_scale
    except Exception as e:
        print(f"[get_or_compute_skater_features] Failed for {skater_name}: {e}")
        return None, None


def resample_to_length(arr, length=100):
    """Resamples a 1D series to a fixed length via linear interpolation over
    normalized [0, 1] position, so two skaters' clips of different lengths
    (and different stride timing) become directly comparable position-by-
    position, e.g. '20% through the sampled window'."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.zeros(length)
    if len(arr) == 1:
        return np.full(length, arr[0])
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, length)
    return np.interp(x_new, x_old, arr)


def dtw_distance(a, b):
    """Classic dynamic time warping distance between two equal-or-different
    length 1D sequences, normalized by the warping path length so it's
    comparable across feature pairs of different raw magnitude/length."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            d = abs(ai - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack to measure path length for normalization
    i, j = n, m
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choices = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
            step = int(np.argmin(choices))
            if step == 0:
                i, j = i - 1, j - 1
            elif step == 1:
                i -= 1
            else:
                j -= 1

    return cost[n, m] / max(path_len, 1)


def _zscore(series):
    std = np.std(series)
    if std < 1e-8:
        return series - np.mean(series)
    return (series - np.mean(series)) / std


def compute_cross_skater_comparison(df_ref, df_target, resample_length=100):
    """Compares two skaters' feature DataFrames (as returned by
    process_skating_video_multivariate) across all columns in
    FEATURE_COLUMNS.

    Each feature is:
      1. Resampled to `resample_length` points (normalized stride-cycle
         position, not raw frame index -- makes differing clip lengths
         comparable).
      2. Z-scored (zero mean, unit variance) so features with different
         natural scales (e.g. knee angle in degrees vs. bone-scaled
         position in normalized units) contribute comparably to the
         combined error, instead of one feature dominating purely because
         of its numeric range.
      3. Compared via per-position absolute difference (for the plotted
         error curve) AND via DTW distance (for the scalar summary,
         which tolerates timing/phase misalignment between the two clips).

    Returns (error_curves: dict[label -> np.ndarray of length
    resample_length], dtw_scores: dict[label -> float]).
    """
    error_curves = {}
    dtw_scores = {}

    for col, label in FEATURE_COLUMNS.items():
        if col not in df_ref.columns or col not in df_target.columns:
            continue

        ref_series = df_ref[col].values
        target_series = df_target[col].values
        if len(ref_series) < 5 or len(target_series) < 5:
            continue

        ref_resampled = resample_to_length(ref_series, resample_length)
        target_resampled = resample_to_length(target_series, resample_length)

        ref_z = _zscore(ref_resampled)
        target_z = _zscore(target_resampled)

        error_curves[label] = np.abs(ref_z - target_z)
        dtw_scores[label] = dtw_distance(ref_z, target_z)

    return error_curves, dtw_scores


def summarize_comparison(dtw_scores):
    """Turns per-feature DTW distances into a couple of headline numbers.

    similarity_score is a bounded, monotonically-decreasing-with-error
    convenience metric for display (100 = identical normalized motion
    pattern, lower = more divergent). It is NOT a trained classifier's
    accuracy -- there is no ground truth "correct" cross-subject label
    here, just a kinematic similarity measure.
    """
    if not dtw_scores:
        return {"mean_dtw_distance": None, "similarity_score": None}

    mean_dtw = float(np.mean(list(dtw_scores.values())))
    similarity_score = round(100.0 / (1.0 + mean_dtw), 1)
    return {
        "mean_dtw_distance": round(mean_dtw, 4),
        "similarity_score": similarity_score,
    }