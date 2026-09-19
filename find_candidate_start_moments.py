"""
Phase 4a: finds candidate START-PHASE moments in each skater's video by
detecting real spikes in hip acceleration, rather than assuming "first N
frames = start" (an unverified assumption used only as a math sanity
check in start_phase_features.py's test run).

A genuine explosive push-off should show a distinctive acceleration
signature -- this scans for it algorithmically, then saves a frame image
at each candidate moment so you can visually confirm whether it's a real
start or a false positive (e.g. a stride push-off mid-race can also spike
acceleration, so this is a candidate-finder, not a certainty).

Usage:
    python -m find_candidate_start_moments
"""

import os
import cv2
import numpy as np
from scipy.signal import find_peaks

from run_bone_scaling_ablation import SKATER_VIDEOS, get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

OUTPUT_DIR = "start_moment_candidates"
TOP_N_CANDIDATES_PER_SKATER = 3


def find_candidates_for_skater(skater_name, video_rel_path, velocity_rest_threshold=None,
                                search_start_sec=None, search_end_sec=None, fps=30.0):
    df = get_skater_features(skater_name, video_rel_path, "scaled")
    if df is None:
        print(f"  [SKIP] No features available for {skater_name}")
        return []

    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    accel = df["hip_acceleration"].fillna(0).values
    accel_abs = np.abs(accel)
    velocity = df["hip_velocity"].fillna(0).values

    if velocity_rest_threshold is None:
        velocity_rest_threshold = np.percentile(velocity[velocity > 0], 20)

    # NEW: if an explicit time window is given (in seconds), search there
    # instead of assuming "first 20% of the clip". Use this when you've
    # already visually identified roughly where a start happens.
    if search_start_sec is not None and search_end_sec is not None:
        frame_col = df["frame"].values
        start_frame_target = search_start_sec * fps
        end_frame_target = search_end_sec * fps
        window_mask = (frame_col >= start_frame_target) & (frame_col <= end_frame_target)
        window_indices = np.where(window_mask)[0]
        if len(window_indices) < 5:
            print(f"  [SKIP] Not enough frames in the {search_start_sec}-{search_end_sec}s window for {skater_name}")
            return []
        search_start_idx, search_end_idx = window_indices[0], window_indices[-1] + 1
        print(f"  Searching explicit window: {search_start_sec}s-{search_end_sec}s "
              f"(frames {int(start_frame_target)}-{int(end_frame_target)})")
    else:
        search_end_idx = int(len(df) * 0.2)
        search_start_idx = 0

    accel_abs_windowed = accel_abs[search_start_idx:search_end_idx]

    if len(accel_abs_windowed) < 10:
        print(f"  [SKIP] Not enough frames in search window for {skater_name}")
        return []

    peaks_local, properties = find_peaks(accel_abs_windowed, distance=10, prominence=np.std(accel_abs_windowed) * 1.5)
    peaks = peaks_local + search_start_idx  # convert back to full-df indices

    if len(peaks) == 0:
        print(f"  No significant acceleration spikes found in first 20% of {skater_name}'s clip")
        return []

    # NEW: filter to only peaks where velocity was near-rest shortly BEFORE
    # the spike. This is what distinguishes a genuine start (stationary,
    # then explodes) from a mid-race cornering push-off (already moving,
    # then pushes harder) -- both can spike acceleration, but only a real
    # start should show a low-velocity runway into that spike.
    genuine_start_peaks = []
    rejected_as_cornering = []
    for peak_idx in peaks:
        pre_window_start = max(0, peak_idx - 10)
        velocity_before = velocity[pre_window_start:peak_idx].mean() if peak_idx > pre_window_start else velocity[peak_idx]
        if velocity_before <= velocity_rest_threshold:
            genuine_start_peaks.append(peak_idx)
        else:
            rejected_as_cornering.append((peak_idx, velocity_before))

    if rejected_as_cornering:
        print(f"  Rejected {len(rejected_as_cornering)} spike(s) as likely mid-race push-offs "
              f"(velocity before spike above rest threshold {velocity_rest_threshold:.5f}):")
        for idx, v in rejected_as_cornering:
            print(f"    frame {int(df.iloc[idx]['frame'])}: velocity_before={v:.5f}")

    if len(genuine_start_peaks) == 0:
        print(f"  No candidates passed the near-rest-before-spike filter for {skater_name}")
        return []

    genuine_start_peaks = np.array(genuine_start_peaks)
    prominences_all = properties["prominences"]
    peak_to_prominence = dict(zip(peaks, prominences_all))
    genuine_prominences = np.array([peak_to_prominence[p] for p in genuine_start_peaks])

    top_indices = np.argsort(genuine_prominences)[::-1][:TOP_N_CANDIDATES_PER_SKATER]
    top_peaks = genuine_start_peaks[top_indices]

    candidates = []
    for peak_idx in top_peaks:
        frame_num = int(df.iloc[peak_idx]["frame"])
        torso_lean = df.iloc[peak_idx]["torso_lean_angle_deg"]
        accel_val = df.iloc[peak_idx]["hip_acceleration"]
        vel_before = velocity[max(0, peak_idx - 10):peak_idx].mean()
        candidates.append({
            "skater": skater_name,
            "frame": frame_num,
            "torso_lean_angle_deg": torso_lean,
            "hip_acceleration": accel_val,
            "velocity_before": vel_before,
        })

    return candidates


def save_candidate_frame_image(video_rel_path, frame_num, output_path):
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_rel_path)
    cap = cv2.VideoCapture(full_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_path, frame)
        return True
    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 4a: Finding candidate start-moment frames via acceleration spikes")
    print("=" * 70)

    all_candidates = []
    for skater_name, video_rel_path in SKATER_VIDEOS.items():
        print(f"\n{skater_name}...")
        candidates = find_candidates_for_skater(skater_name, video_rel_path)
        for c in candidates:
            print(f"  Candidate: frame {c['frame']}, torso_lean={c['torso_lean_angle_deg']:.1f}°, "
                  f"accel={c['hip_acceleration']:.4f}, velocity_before={c['velocity_before']:.5f}")
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in skater_name)
            img_path = os.path.join(OUTPUT_DIR, f"{safe_name}_frame{c['frame']}.jpg")
            if save_candidate_frame_image(video_rel_path, c["frame"], img_path):
                print(f"    Saved frame image -> {img_path}")
                c["image_path"] = img_path
        all_candidates.extend(candidates)

    print("\n" + "=" * 70)
    print(f"Done. Saved {len(all_candidates)} candidate frame images to {OUTPUT_DIR}/")
    print("=" * 70)
    print("\nNEXT STEP: open a few of these images and visually confirm whether")
    print("they show a genuine crouched start position, or a false positive")
    print("(e.g. a strong mid-race stride push-off, which can also spike")
    print("acceleration). Only candidates you visually confirm as real starts")
    print("should be used to define the start-phase boundary for the full")
    print("three-condition ablation.")
    print(f"\nQuick way to open them all: Invoke-Item .\\{OUTPUT_DIR}\\")


if __name__ == "__main__":
    main()
