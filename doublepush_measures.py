"""
Phase 7 (10/19): double-push measures for labelled stretches of video.

For each stretch in doublepush/style_labels.csv (video, start_frame,
end_frame, label), MediaPipe pose runs on every frame (VIDEO mode, restarted
at camera cuts) and these are computed:

Detection layer (double push vs normal stride)
  cross_depth   how far a foot gets to the OTHER side of the hip midline
                (95th percentile over the stretch, the larger of the two
                feet), in leg lengths. Double push brings the foot in under
                the body (the underpush); a normal push goes straight out.
  cross_frac    share of frames with a foot past the midline by > 0.1 leg.

Quality layer (good vs flawed double push)
  foot_jerk     roughness of the feet's sideways paths (mean absolute second
                difference, per second^2, leg lengths): smooth S-curve vs
                jerky, "wild" movement.
  hip_bob       up-down movement of the hips (std after removing slow drift,
                leg lengths): pressure dropped and regained.
  upper_sway    side-to-side movement of the shoulders relative to the hips
                (std, leg lengths); only frames with both shoulders visible
                (visibility > 0.5), and blank if fewer than half are.

Scale: median hip-to-ankle length in the stretch (10/18: torso length was
unreliable when the upper body was out of frame). Foot = midpoint of heel
and toe. Front or back view is assumed for the sideways measures.

Output: doublepush/measures.csv (one row per stretch),
        doublepush/frames_<n>.csv (per-frame values, for plots and checks).

Usage:
    python -m doublepush_measures
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from detect_cuts import frame_diffs, cuts_from_diffs

LABELS = "doublepush/style_labels.csv"
MODEL = "pose_landmarker_lite.task"
OUT = "doublepush/measures.csv"
VIS = 0.5


def track(video, a, b):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = cap.get(3), cap.get(4)
    cuts = set(cuts_from_diffs(frame_diffs(video, a, b)))
    opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    lmk, t0, rows = vision.PoseLandmarker.create_from_options(opts), a, []
    for f in range(a, b + 1):
        ok, img = cap.read()
        if not ok:
            break
        if f in cuts:
            lmk.close(); lmk = vision.PoseLandmarker.create_from_options(opts); t0 = f
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), int((f - t0) * 1000 / fps))
        if not res.pose_landmarks:
            continue
        lm = res.pose_landmarks[0]
        X = lambda i: lm[i].x * W
        Y = lambda i: lm[i].y * H
        hip_x, hip_y = (X(23) + X(24)) / 2, (Y(23) + Y(24)) / 2
        leg = (np.hypot(X(23) - X(27), Y(23) - Y(27)) + np.hypot(X(24) - X(28), Y(24) - Y(28))) / 2
        rows.append({"frame": f, "cut_seq": sum(c <= f for c in cuts), "leg_px": leg,
                     "hip_x": hip_x, "hip_y": hip_y,
                     "lf": (X(29) + X(31)) / 2 - hip_x, "rf": (X(30) + X(32)) / 2 - hip_x,
                     "sh_dx": (X(11) + X(12)) / 2 - hip_x,
                     "sh_vis": min(lm[11].visibility, lm[12].visibility),
                     "foot_vis": min(lm[i].visibility for i in (29, 30, 31, 32))})
    lmk.close()
    return pd.DataFrame(rows), fps


def measures(d, fps):
    if len(d) < 15:
        return {"frames": len(d)}
    s = d.leg_px.median()
    lf, rf = d.lf / s, d.rf / s
    # which foot is on which side: in a front view the skater's left foot appears on the image right
    side = np.sign(lf.median() - rf.median()) or 1.0
    lcross, rcross = (-side * lf).clip(lower=0), (side * rf).clip(lower=0)
    out = {"frames": len(d), "leg_px": round(s, 1),
           "cross_depth": max(lcross.quantile(0.95), rcross.quantile(0.95)),
           "cross_frac": float(((lcross > 0.1) | (rcross > 0.1)).mean())}
    jerk = []
    for _, g in d.groupby("cut_seq"):
        if len(g) < 5:
            continue
        for col in ("lf", "rf"):
            v = (g[col] / s).values
            jerk.append(np.abs(np.diff(v, 2)).mean() * fps ** 2)
    out["foot_jerk"] = float(np.mean(jerk)) if jerk else np.nan
    hy = d.hip_y / s
    out["hip_bob"] = float((hy - hy.rolling(int(fps), center=True, min_periods=3).mean()).std())
    vis = d[d.sh_vis > VIS]
    out["upper_vis_frac"] = len(vis) / len(d)
    out["upper_sway"] = float((vis.sh_dx / s).std()) if len(vis) >= len(d) / 2 else np.nan
    return out


def main():
    lab = pd.read_csv(LABELS)
    res = []
    for n, r in enumerate(lab.itertuples()):
        d, fps = track(r.video, int(r.start_frame), int(r.end_frame))
        d.to_csv(f"doublepush/frames_{n:02d}.csv", index=False)
        m = measures(d, fps)
        res.append({"n": n, "label": r.label, "stretch": r.skater_note, **m})
        print(f"[{n}] {r.label:20s} {r.skater_note[:40]:40s} " +
              "  ".join(f"{k}={v:.2f}" for k, v in m.items() if isinstance(v, float)), flush=True)
    pd.DataFrame(res).to_csv(OUT, index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
