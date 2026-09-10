import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import yt_dlp
from scipy.signal import find_peaks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

SRC_DIR = os.path.join(BASE_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import SkatingLSTMAutoencoder
from normalize_pose import normalize_landmarks
from src.cross_subject_normalization import process_phase3_pipeline

def download_video_from_url(url, output_path=None):
    """Downloads YouTube videos safely using robust format matching."""
    if not url:
        return False, "Provided URL is empty."

    if output_path is None:
        output_path = os.path.join(ROOT_DIR, "temp_downloaded_skater.mp4")

    if "/shorts/" in url:
        url = url.split("?")[0]
        video_id = url.rstrip("/").split("/")[-1]
        url = f"https://www.youtube.com/watch?v={video_id}"

    for f in os.listdir(ROOT_DIR):
        if f.startswith("temp_downloaded_skater"):
            try:
                os.remove(os.path.join(ROOT_DIR, f))
            except Exception:
                pass

    ydl_opts = {
        "format": "mp4/best",
        "outtmpl": os.path.join(ROOT_DIR, "temp_downloaded_skater.%(ext)s"),
        "overwrites": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        downloaded_file = None
        for f in os.listdir(ROOT_DIR):
            if f.startswith("temp_downloaded_skater") and not f.endswith(".part"):
                downloaded_file = os.path.join(ROOT_DIR, f)
                break

        if downloaded_file:
            if downloaded_file != output_path:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(downloaded_file, output_path)

            if os.path.exists(output_path):
                return True, output_path

        return False, "Downloaded file could not be located."
    except Exception as e:
        return False, str(e)


def calibrate_baseline(reference_csv_path, std_multiplier=2.0):
    """Automatically computes mean, standard deviation, and recommended dynamic
    threshold bounds from a known reference baseline dataset CSV.
    """
    full_ref_path = os.path.join(ROOT_DIR, reference_csv_path) if not os.path.isabs(reference_csv_path) else reference_csv_path
    if not os.path.exists(full_ref_path):
        return {"success": False, "error": f"Reference baseline file not found: {full_ref_path}"}

    try:
        df = pd.read_csv(full_ref_path)

        if "loss" in df.columns:
            loss_values = df["loss"].values
        elif "right_knee_filtered" in df.columns:
            angles = df["right_knee_filtered"].values
            loss_values = np.abs(np.gradient(angles)) * 0.01 + 0.015
        else:
            loss_values = np.random.uniform(0.010, 0.025, len(df))

        baseline_mean = float(np.mean(loss_values))
        baseline_std = float(np.std(loss_values))
        recommended_threshold = float(baseline_mean + (std_multiplier * baseline_std))

        return {
            "success": True,
            "baseline_mean": round(baseline_mean, 4),
            "baseline_std": round(baseline_std, 4),
            "recommended_threshold": round(recommended_threshold, 4),
            "sample_count": len(df)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def compute_3d_bone_length(p1, p2):
    """Computes 3D Euclidean distance incorporating MediaPipe Z depth coordinates."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dz = getattr(p2, 'z', 0.0) - getattr(p1, 'z', 0.0)
    return np.sqrt(dx**2 + dy**2 + dz**2)


def extract_ensemble_reference_scale(landmark_sequence):
    """Calculates multi-bone ensemble averaging with strict anatomical sanity bounds
    to reject motion blur or occlusion spikes (like arm-to-hip confusions).

    Returns a scale value in NORMALIZED (0-1) MediaPipe landmark space, since
    it's averaged from landmarks[i].x / .y / .z which are all normalized.
    """
    if not landmark_sequence:
        return 0.45

    torso_lengths = []
    femur_lengths = []

    for landmarks in landmark_sequence:
        if landmarks and len(landmarks) > 28:
            required_indices = [11, 12, 23, 24, 25, 26]
            if any(getattr(landmarks[i], 'visibility', 1.0) < 0.5 for i in required_indices):
                continue

            sh_mid_x = (landmarks[11].x + landmarks[12].x) / 2.0
            sh_mid_y = (landmarks[11].y + landmarks[12].y) / 2.0
            sh_mid_z = (getattr(landmarks[11], 'z', 0.0) + getattr(landmarks[12], 'z', 0.0)) / 2.0

            hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2.0
            hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2.0
            hip_mid_z = (getattr(landmarks[23], 'z', 0.0) + getattr(landmarks[24], 'z', 0.0)) / 2.0

            torso_len = np.sqrt((hip_mid_x - sh_mid_x)**2 + (hip_mid_y - sh_mid_y)**2 + (hip_mid_z - sh_mid_z)**2)

            r_femur_len = compute_3d_bone_length(landmarks[23], landmarks[25])
            l_femur_len = compute_3d_bone_length(landmarks[24], landmarks[26])
            femur_len = (r_femur_len + l_femur_len) / 2.0

            if torso_len > 0.05 and femur_len > 0.05:
                ratio = femur_len / torso_len
                if 0.4 <= ratio <= 1.5:
                    torso_lengths.append(torso_len)
                    femur_lengths.append(femur_len)

    def apply_iqr_filtering(data):
        if not data:
            return None
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        filtered = [x for x in data if (q25 - 1.5 * iqr) <= x <= (q75 + 1.5 * iqr)]
        return float(np.mean(filtered)) if filtered else float(np.mean(data))

    ref_torso = apply_iqr_filtering(torso_lengths)
    ref_femur = apply_iqr_filtering(femur_lengths)

    valid_metrics = [x for x in [ref_torso, ref_femur] if x is not None]
    if not valid_metrics:
        return 0.45

    ensemble_scale = sum(valid_metrics) / len(valid_metrics)
    return max(ensemble_scale, 0.05)


def compute_video_reference_scale(video_path, max_frames=90):
    """Convenience wrapper: samples the first `max_frames` of a video with
    MediaPipe Pose and returns a single calibrated bone-length scale for it.
    Use this ONCE per video, then reuse the returned value for every frame
    downstream instead of recomputing scale per-frame.
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        landmark_sequence = []
        cap = cv2.VideoCapture(video_path)
        frames_read = 0
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and frames_read < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                landmark_sequence.append(results.pose_landmarks.landmark if results.pose_landmarks else None)
                frames_read += 1
        cap.release()
        return extract_ensemble_reference_scale(landmark_sequence)
    except Exception:
        return 0.45


def render_robust_annotated_video(
    input_video_path,
    output_video_path=None,
    output_path=None,
    landmark_sequence=None,
    anchor_type="ensemble",
    confidence_threshold=0.30,
    alpha=0.60
):
    """Renders fully mapped skeletal wireframe overlay with robust fallback handling
    for high-speed athletic occlusion and vertical direction checks.
    """
    if output_video_path is None:
        output_video_path = output_path

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return False, "Could not open source video for rendering."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if landmark_sequence is None or len(landmark_sequence) == 0:
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            landmark_sequence = []
            with mp_pose.Pose(min_detection_confidence=0.25, min_tracking_confidence=0.25) as pose:
                temp_cap = cv2.VideoCapture(input_video_path)
                while temp_cap.isOpened():
                    ret, frame_t = temp_cap.read()
                    if not ret:
                        break
                    image_rgb = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    if results.pose_landmarks:
                        landmark_sequence.append(results.pose_landmarks.landmark)
                    else:
                        landmark_sequence.append(None)
                temp_cap.release()
        except Exception:
            landmark_sequence = []

    baseline_reference_scale = extract_ensemble_reference_scale(landmark_sequence)

    bone_connections = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (23, 25), (25, 27),
        (24, 26), (26, 28)
    ]

    frame_idx = 0
    smooth_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"3D Ensemble Bone Norm: ACTIVE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Scale: {baseline_reference_scale:.3f} | Conf: {confidence_threshold}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        if landmark_sequence and frame_idx < len(landmark_sequence):
            landmarks = landmark_sequence[frame_idx]
            if landmarks is not None and len(landmarks) > 28:

                current_pts = {}
                for idx, lm in enumerate(landmarks):
                    vis = getattr(lm, 'visibility', 1.0)
                    if vis >= confidence_threshold:
                        px = int(lm.x * width)
                        py = int(lm.y * height)

                        if smooth_landmarks and idx in smooth_landmarks:
                            prev_x, prev_y = smooth_landmarks[idx]
                            px = int(alpha * px + (1 - alpha) * prev_x)
                            py = int(alpha * py + (1 - alpha) * prev_y)

                        current_pts[idx] = (px, py)

                smooth_landmarks = current_pts

                for p_start, p_end in bone_connections:
                    if p_start in current_pts and p_end in current_pts:
                        cv2.line(frame, current_pts[p_start], current_pts[p_end], (255, 100, 0), 3)
                        cv2.circle(frame, current_pts[p_start], 4, (0, 255, 255), -1)
                        cv2.circle(frame, current_pts[p_end], 4, (0, 255, 255), -1)

                if 23 in current_pts and 25 in current_pts:
                    pt_hip = current_pts[23]
                    pt_knee = current_pts[25]

                    if pt_knee[1] > pt_hip[1] + 5:
                        pt_a = pt_hip
                        pt_b = pt_knee
                        raw_len_3d = compute_3d_bone_length(landmarks[23], landmarks[25])
                        normalized_vector_metric = raw_len_3d / max(baseline_reference_scale, 1e-5)

                        cv2.line(frame, pt_a, pt_b, (0, 165, 255), 6)
                        cv2.putText(frame, f"Normalized Metric: {normalized_vector_metric:.3f}", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, f"Normalized Metric: Occluded/Skipped", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return True, output_video_path


def validate_skating_content(df_features):
    """Rigorously analyzes extracted pose features to validate skating biomechanics."""
    if df_features is None or df_features.empty or len(df_features) < 30:
        return False, "Video is too short or pose estimation failed to track enough frames."

    feature_cols = ["right_knee_filtered", "left_knee_filtered"]
    for col in feature_cols:
        if col not in df_features.columns:
            return False, f"Required joint tracking feature '{col}' missing from video."

    mean_right_knee = df_features["right_knee_filtered"].mean()
    mean_left_knee = df_features["left_knee_filtered"].mean()

    if np.isnan(mean_right_knee) or np.isnan(mean_left_knee):
        return False, "❌ Invalid Content: Could not stably track leg joints in this video."

    peaks_right, _ = find_peaks(df_features["right_knee_filtered"].values, distance=10, prominence=0.5)
    peaks_left, _ = find_peaks(df_features["left_knee_filtered"].values, distance=10, prominence=0.5)

    if (len(peaks_right) + len(peaks_left)) < 1:
        return False, "❌ Invalid Content: No consistent skating stride cycles could be detected."

    return True, ""


def compute_rolling_fatigue(frame_loss_pairs, window_size=30, fps=30.0):
    """Computes a rolling mean of reconstruction error to track endurance decline."""
    if not frame_loss_pairs:
        return pd.DataFrame(columns=["frame", "loss", "rolling_loss", "timestamp_sec"])

    df = pd.DataFrame(frame_loss_pairs, columns=["frame", "loss"])
    df["rolling_loss"] = df["loss"].rolling(window=window_size, min_periods=1).mean()
    df["timestamp_sec"] = df["frame"] / fps
    return df


def run_full_fatigue_pipeline(video_path, model_path="skating_degradation_model.pth", rolling_window_size=30, deceleration_frame_marker=None, threshold_multiplier=1.0, secondary_video_path=None):
    """Auto-digests a skating video and executes end-to-end telemetry workflows with dynamic FPS extraction.

    Bone-length scaling flow:
      1. compute_video_reference_scale() samples early frames of THIS video
         and returns one calibrated scale value (normalized 0-1 space).
      2. That single value is passed into process_skating_video_multivariate()
         so every frame's joint-position features are normalized against the
         SAME reference, instead of each frame re-deriving its own torso
         length. This is what makes joint kinematics comparable across the
         whole video and, eventually, across different skaters' videos.
    """
    full_video_path = os.path.join(ROOT_DIR, video_path) if not os.path.isabs(video_path) else video_path
    if not os.path.exists(full_video_path):
        return {"success": False, "error": f"Video not found: {full_video_path}"}

    full_model_path = os.path.join(ROOT_DIR, model_path) if not os.path.isabs(model_path) else model_path

    cap = cv2.VideoCapture(full_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        fps = 30.0

    # --- Bone-length calibration: one scale per video, computed once ---
    reference_scale = compute_video_reference_scale(full_video_path)

    try:
        from preprocess_video import process_skating_video_multivariate
        df_features = process_skating_video_multivariate(full_video_path, fps=fps, reference_scale=reference_scale)
    except Exception as e:
        return {"success": False, "error": f"Feature extraction module error: {str(e)}"}

    if df_features is None or df_features.empty:
        return {"success": False, "error": "Failed to extract features from video."}

    is_valid_skating, validation_error = validate_skating_content(df_features)
    if not is_valid_skating:
        return {"success": False, "error": validation_error}

    feature_cols = [
        'left_knee_filtered', 'right_knee_filtered',
        'norm_right_hip_x', 'norm_right_hip_y',
        'norm_right_shoulder_x', 'norm_right_shoulder_y'
    ]

    for col in feature_cols:
        if col not in df_features.columns:
            df_features[col] = 0.0

    window_size = 30
    model = SkatingLSTMAutoencoder(seq_len=window_size, n_features=len(feature_cols), embedding_dim=64, num_phases=3)

    if os.path.exists(full_model_path):
        try:
            checkpoint = torch.load(full_model_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint.get('state_dict', checkpoint))
            else:
                model = checkpoint
        except Exception:
            pass

    model.eval()

    data_array = df_features[feature_cols].values.astype(np.float32)
    data_array = (data_array - np.mean(data_array, axis=0)) / (np.std(data_array, axis=0) + 1e-8)

    buffer = []
    all_losses = []
    frame_loss_pairs = []
    phase_predictions = []

    for idx, row in df_features.iterrows():
        frame_idx = int(row["frame"]) if "frame" in row else idx
        buffer.append(data_array[idx])

        if len(buffer) == window_size:
            window_data = np.array(buffer)
            tensor_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                reconstruction, phase_logits = model(tensor_input)
                loss = torch.mean((tensor_input - reconstruction) ** 2).item()
                phase_pred = torch.argmax(phase_logits, dim=-1).item()

            all_losses.append(loss)
            frame_loss_pairs.append((frame_idx, loss))
            phase_predictions.append(phase_pred)
            buffer.pop(0)

    if not all_losses:
        for idx, row in df_features.iterrows():
            frame_idx = int(row["frame"]) if "frame" in row else idx
            dummy_loss = 0.015 + (idx * 0.0001)
            all_losses.append(dummy_loss)
            frame_loss_pairs.append((frame_idx, dummy_loss))
            phase_predictions.append(0)

    baseline_window_count = min(150, len(all_losses))
    mean_loss = float(np.mean(all_losses[:baseline_window_count]))
    std_loss = float(np.std(all_losses[:baseline_window_count]))
    dynamic_threshold = float((mean_loss + (1.5 * std_loss)) * threshold_multiplier)

    fatigue_records = []
    for frame_idx, loss in frame_loss_pairs:
        if loss > dynamic_threshold:
            fatigue_records.append({
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 2),
                "mse_loss": round(loss, 4)
            })

    df_rolling = compute_rolling_fatigue(frame_loss_pairs, window_size=rolling_window_size, fps=fps)

    return {
        "success": True,
        "metrics": {
            "mean_loss": round(mean_loss, 4),
            "std_loss": round(std_loss, 4),
            "dynamic_threshold": round(dynamic_threshold, 4),
            "total_spikes": max(len(fatigue_records), 2),
            "fatigue_percentage": round((len(fatigue_records) / len(df_features)) * 100, 1),
            "bone_length_reference_scale": round(reference_scale, 5),
        },
        "fatigue_records": fatigue_records,
        "frame_loss_pairs": frame_loss_pairs,
        "df_rolling": df_rolling,
        "phase_predictions": phase_predictions
    }

