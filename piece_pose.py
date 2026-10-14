"""
Phase 6c (10/12): pose features for pure pieces, measured on the boxed skater.

Phase 5's features (preprocess_video.process_skating_video_multivariate) run
MediaPipe on the WHOLE frame and keep the first person it finds -- fine when
one skater is in frame, but with two skaters it may measure the wrong one.
Here MediaPipe runs on a crop around the piece's tracker box, so the pose is
the boxed skater's. Same model (pose_landmarker_lite) and same feature
formulas as Phase 5, with coordinates converted back to full-frame pixels.

Frame numbers: tracker frames are 0-based; Phase 5 feature frames are 1-based
(frame_count is incremented before the first frame is stored). The output
uses the tracker's 0-based frames, plus `p5_frame` = frame + 1 for comparison.

Check (10/12) against the 8 researcher-verified Inzell Eitrem segments (one
skater in frame, so Phase 5's whole-frame pose is on him):
  - 6 side-on segments: segment-mean knee angles within ~4 deg of Phase 5,
    torso lean within ~5 deg (one straightaway: 9.6 deg).
  - 2 front-on segments: one agrees; one head-on close-up (7342-7372) is
    clearly wrong in the crop version (knees ~172 deg, lean ~6 deg while
    the skater is visibly crouched) -- use front-on pieces with care.
  - Per-frame differences are 5-8 deg typical in both methods' noise;
    agreement does not depend on box size.
  - The larger pose_landmarker_full model did NOT fix the close-up and
    agreed less with Phase 5, so the lite model is kept for comparability.

Output: dual_tracking/<name>_piece_pose.csv

Usage:
    python -m piece_pose --video data/inzell_2026_5000m.mp4 --name inzell_full_race
    python -m piece_pose ... --pieces 17,23,58      (only these piece numbers)
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from utils.biomechanics_utils import calculate_angle, butter_lowpass_filter

MODEL = "pose_landmarker_lite.task"   # same model as Phase 5
PAD_X = 0.30                          # crop margin, fraction of box width each side
PAD_Y = 0.15                          # ... and of box height top/bottom
MIN_CROP_H = 256                      # small crops are upscaled to at least this height


def crop_box(frame, x1, y1, x2, y2):
    h_img, w_img = frame.shape[:2]
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - PAD_X * w)), int(max(0, y1 - PAD_Y * h))
    cx2, cy2 = int(min(w_img, x2 + PAD_X * w)), int(min(h_img, y2 + PAD_Y * h))
    crop = frame[cy1:cy2, cx1:cx2]
    scale = max(1.0, MIN_CROP_H / max(1, crop.shape[0]))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, cx1, cy1, scale


def features(lm, cx1, cy1, scale, cw, ch):
    """Landmarks (normalised to the crop) -> Phase 5 per-frame values in full-frame pixels.
    Positions are kept in pixels relative to the hip midpoint here; main() divides
    them by one scale per piece (see PIECE_SCALE)."""
    def px(i):
        return np.array([lm[i].x * cw / scale + cx1, lm[i].y * ch / scale + cy1], dtype=np.float32)

    def vis(i):
        return float(getattr(lm[i], "visibility", 1.0))

    l_sh, r_sh, l_el, r_el, l_wr, r_wr = px(11), px(12), px(13), px(14), px(15), px(16)
    l_hip, r_hip, l_kn, r_kn, l_an, r_an = px(23), px(24), px(25), px(26), px(27), px(28)
    mid_hip, mid_sh = (l_hip + r_hip) / 2, (l_sh + r_sh) / 2
    rel = {"right_hip": r_hip, "right_shoulder": r_sh, "left_hip": l_hip, "left_shoulder": l_sh,
           "right_elbow": r_el, "left_elbow": l_el, "right_ankle": r_an, "left_ankle": l_an}
    out = {"right_knee": calculate_angle(r_hip.tolist(), r_kn.tolist(), r_an.tolist()),
           "left_knee": calculate_angle(l_hip.tolist(), l_kn.tolist(), l_an.tolist()),
           "right_elbow_angle": calculate_angle(r_sh.tolist(), r_el.tolist(), r_wr.tolist()),
           "left_elbow_angle": calculate_angle(l_sh.tolist(), l_el.tolist(), l_wr.tolist()),
           "frame_torso_length_px": float(np.linalg.norm(mid_sh - mid_hip)),
           "mid_hip_x": float(mid_hip[0]), "mid_hip_y": float(mid_hip[1]),
           "right_elbow_visibility": vis(14), "left_elbow_visibility": vis(13),
           "right_wrist_visibility": vis(16), "left_wrist_visibility": vis(15),
           "knee_vis": min(vis(25), vis(26)), "ankle_vis": min(vis(27), vis(28))}
    for k, v in rel.items():
        out[f"px_{k}_x"], out[f"px_{k}_y"] = float(v[0] - mid_hip[0]), float(v[1] - mid_hip[1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--pieces", help="comma-separated piece numbers (default: all)")
    a = p.parse_args()

    tracks = pd.read_csv(f"dual_tracking/{a.name}_tracks.csv")
    pieces = pd.read_csv(f"dual_tracking/{a.name}_pieces.csv")
    if a.pieces:
        pieces = pieces[pieces.piece_no.isin([int(x) for x in a.pieces.split(",")])]
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.VIDEO)
    out = []
    for r in pieces.sort_values("first_frame").itertuples():
        boxes = tracks[(tracks.track_id == r.track_id) & (tracks.frame >= r.first_frame)
                       & (tracks.frame <= r.last_frame)].set_index("frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.first_frame)
        rows = []
        # a fresh landmarker per piece, so tracking never carries over between pieces
        with vision.PoseLandmarker.create_from_options(opts) as lmk:
            for f in range(r.first_frame, r.last_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if f not in boxes.index:
                    continue
                b = boxes.loc[f]
                crop, cx1, cy1, s = crop_box(frame, b.x1, b.y1, b.x2, b.y2)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                res = lmk.detect_for_video(img, int(f * 1000 / fps))
                if not res.pose_landmarks:
                    continue
                feat = features(res.pose_landmarks[0], cx1, cy1, s, crop.shape[1], crop.shape[0])
                if feat["frame_torso_length_px"] < 10.0:   # same anomaly filter as Phase 5
                    continue
                rows.append({"piece_no": r.piece_no, "frame": f, "p5_frame": f + 1,
                             "box_h": b.y2 - b.y1, **feat})
        if len(rows) >= 10:
            d = pd.DataFrame(rows)
            for side in ("right", "left"):
                d[f"{side}_knee_filtered"] = butter_lowpass_filter(d[f"{side}_knee"].values, cutoff_freq=5.0, sample_rate=fps)
            # PIECE_SCALE: positions divided by the piece's median torso length,
            # one value per piece, stored as scale_used_px -- so Phase 5's
            # implausible-frame filter and per-segment rescale work unchanged
            d["scale_used_px"] = d["frame_torso_length_px"].median()
            for c in [c for c in d.columns if c.startswith("px_")]:
                d["norm_" + c[3:]] = d[c] / d["scale_used_px"]
            d["torso_lean_angle_deg"] = np.degrees(np.abs(np.arctan2(
                d.norm_right_shoulder_x - d.norm_right_hip_x, -(d.norm_right_shoulder_y - d.norm_right_hip_y))))
            out.append(d)
        n_frames = r.last_frame - r.first_frame + 1
        print(f"P{r.piece_no:<3d} frames {r.first_frame}-{r.last_frame}: pose found in {len(rows)}/{n_frames}", flush=True)
    cap.release()
    if out:
        pd.concat(out).to_csv(f"dual_tracking/{a.name}_piece_pose.csv", index=False)
        print(f"-> dual_tracking/{a.name}_piece_pose.csv")


if __name__ == "__main__":
    main()
