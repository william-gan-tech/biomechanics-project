"""
Phase 5f: Fatigue-Linked Form Degradation.

Question: does a skater's form measurably change between EARLY and LATE
laps of the same race, within the same technique phase -- and does the
Phase 1/3 fatigue autoencoder pick up the same change?

Requires full-race footage labeled in labeled_segments.csv with the lap in
the notes, e.g.  "lap 2, identity-verified (0.87)...". Only corner and
straightaway segments are used, and early is only ever compared with late
WITHIN the same phase (corners change form for reasons unrelated to fatigue).
Early = first third of the skater's labeled laps, late = last third.

ASSUMPTION, stated as in Phase 3: late-race = more fatigued. This is a proxy,
not verified ground-truth fatigue (no lactate/HR data).

Three parts:
  A. Form metrics (interpretable): every Tier 1 metric from
     compare_to_elite_reference (with the 9/28 per-segment rescale and
     per-second units) plus stride rhythm (5d), early vs late per phase.
  B. Autoencoder: reconstruction loss of the saved Phase 1/3 model
     (skating_degradation_model.pth) on early vs late segments, standardized
     with the skater's EARLY-race stats only (Phase 3 rule: never let the
     "fatigued" data set its own baseline).
  C. Camera confound check for B: the model's inputs include hip/shoulder
     POSITIONS on the fixed per-video scale -- shown 9/28 to shift with camera
     zoom. Broadcast cameras zoom constantly during a race, so B's loss is
     correlated against on-screen body size. A strong correlation means B is
     at least partly measuring the camera, not the skater.

Usage:
    python -m validate_fatigue_form_degradation
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from scipy import stats

from compare_to_elite_reference import load_labeled_segments, get_segment_frames, METRICS
from stride_rhythm import analyze_segment

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from model import SkatingLSTMAutoencoder  # noqa: E402

PHASES = ["corner", "straightaway"]
MODEL_PATH = os.path.join(ROOT_DIR, "skating_degradation_model.pth")
AE_FEATURES = [  # must match pipeline_engine.run_full_fatigue_pipeline
    "left_knee_filtered", "right_knee_filtered",
    "norm_right_hip_x", "norm_right_hip_y",
    "norm_right_shoulder_x", "norm_right_shoulder_y",
]
WINDOW = 30
RHYTHM_METRICS = ["stride_period_s", "rhythm_cv"]
OUTPUT_PATH = "phase5f_fatigue_results.csv"
LAP_RE = re.compile(r"\blap\s*(\d+)", re.IGNORECASE)


def lap_segments():
    """Corner/straightaway segments with 'lap N' in the notes, tagged early/late."""
    segs = []
    for row in load_labeled_segments():
        if row["phase"] not in PHASES:
            continue
        m = LAP_RE.search(row.get("notes", ""))
        if m:
            segs.append({**row, "lap": int(m.group(1))})
    if not segs:
        return pd.DataFrame()
    df = pd.DataFrame(segs)

    # Thirds of each skater's own labeled lap range
    lo = df.groupby("skater")["lap"].transform("min")
    hi = df.groupby("skater")["lap"].transform("max")
    third = (hi - lo) / 3
    df["stage"] = np.where(df["lap"] <= lo + third, "early",
                           np.where(df["lap"] >= hi - third, "late", "middle"))
    return df[df["stage"] != "middle"]


def part_a_form(segs):
    rows = []
    for _, r in segs.iterrows():
        frames = get_segment_frames(r)
        if frames is None:
            continue
        rec = {"skater": r["skater"], "phase": r["phase"], "stage": r["stage"], "lap": r["lap"],
               "segment": f"{r['start_frame']}-{r['end_frame']}"}
        for m in METRICS:
            rec[m] = frames[m].mean() if m in frames.columns else np.nan
        rhythm, why = analyze_segment(r)
        for m in RHYTHM_METRICS:
            rec[m] = rhythm.get(m, np.nan) if (rhythm is not None and why is None) else np.nan
        rec["_frames"] = frames
        rows.append(rec)
    return rows


def summarize_a(rows):
    all_metrics = METRICS + RHYTHM_METRICS
    out = []
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_frames"} for r in rows])
    for (skater, phase), g in df.groupby(["skater", "phase"]):
        e, l = g[g["stage"] == "early"], g[g["stage"] == "late"]
        if e.empty or l.empty:
            continue
        # frame-level effect size for per-frame metrics (autocorrelated -> trust
        # direction and size, not p)
        ef = pd.concat([r["_frames"] for r in rows if r["skater"] == skater and r["phase"] == phase and r["stage"] == "early"])
        lf = pd.concat([r["_frames"] for r in rows if r["skater"] == skater and r["phase"] == phase and r["stage"] == "late"])
        for m in all_metrics:
            early_mean, late_mean = e[m].mean(), l[m].mean()
            rb = np.nan
            if m in ef.columns and m in lf.columns:
                a, b = lf[m].dropna(), ef[m].dropna()
                if len(a) >= 5 and len(b) >= 5:
                    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
                    rb = 2 * u / (len(a) * len(b)) - 1   # +1 = late always higher
            out.append({"skater": skater, "phase": phase, "metric": m,
                        "n_early_segs": len(e), "n_late_segs": len(l),
                        "early": early_mean, "late": late_mean,
                        "pct_change": 100 * (late_mean - early_mean) / abs(early_mean) if early_mean else np.nan,
                        "late_vs_early_effect": rb})
    return pd.DataFrame(out)


def load_model():
    model = SkatingLSTMAutoencoder(seq_len=WINDOW, n_features=len(AE_FEATURES), embedding_dim=64, num_phases=3)
    if not os.path.exists(MODEL_PATH):
        return None
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt.state_dict())
    model.eval()
    return model


def part_b_c_autoencoder(rows):
    model = load_model()
    if model is None:
        print(f"  [SKIP] {MODEL_PATH} not found -- parts B and C skipped")
        return None, None

    windows = []
    for skater in {r["skater"] for r in rows}:
        mine = [r for r in rows if r["skater"] == skater]
        early_frames = pd.concat([r["_frames"] for r in mine if r["stage"] == "early"])
        mu = early_frames[AE_FEATURES].mean().values
        sd = early_frames[AE_FEATURES].std().values + 1e-8
        for r in mine:
            f = r["_frames"]
            arr = ((f[AE_FEATURES].values - mu) / sd).astype(np.float32)
            body = f["frame_torso_length_px"].values if "frame_torso_length_px" in f.columns else np.full(len(f), np.nan)
            for i in range(0, len(arr) - WINDOW + 1):
                x = torch.tensor(arr[i:i + WINDOW]).unsqueeze(0)
                with torch.no_grad():
                    recon, _ = model(x)
                windows.append({"skater": skater, "phase": r["phase"], "stage": r["stage"],
                                "loss": torch.mean((x - recon) ** 2).item(),
                                "torso_px": np.nanmean(body[i:i + WINDOW])})
    w = pd.DataFrame(windows)
    if w.empty:
        return None, None

    b = w.groupby(["skater", "phase", "stage"])["loss"].agg(["mean", "median", "count"]).unstack("stage")
    c = []
    for skater, g in w.groupby("skater"):
        g = g.dropna(subset=["torso_px"])
        if len(g) > 10:
            rho, p = stats.spearmanr(g["torso_px"], g["loss"])
            c.append({"skater": skater, "windows": len(g), "spearman_loss_vs_body_size": rho, "p": p})
    return b, pd.DataFrame(c)


def main():
    segs = lap_segments()
    if segs.empty:
        print("No lap-annotated corner/straightaway segments yet.")
        print("Log full-race segments with the lap in the notes, e.g.:")
        print('  python -m manage_labeled_segments --add --skater "..." --video "data/inzell_2026_5000m.mp4" \\')
        print('      --phase straightaway --start 1200 --end 1275 --notes "lap 2, identity-verified (0.88)" --date 2026-09-28')
        print("Need, per skater: early AND late laps, for corner and/or straightaway.")
        return

    print(f"Using {len(segs)} segments:")
    print(segs.groupby(["skater", "phase", "stage"]).size().to_string())

    rows = part_a_form(segs)
    a = summarize_a(rows)
    pd.set_option("display.width", 220)
    print(f"\n{'='*78}\nA. FORM METRICS: late vs early race (same phase)\n{'='*78}")
    if a.empty:
        print("No skater x phase has both early and late segments.")
    else:
        print(a.round(3).to_string(index=False))
        a.to_csv(OUTPUT_PATH, index=False)
        big = a[a["late_vs_early_effect"].abs() >= 0.3]
        print(f"\nMetrics with at least a moderate late-vs-early effect (|r| >= 0.3): {len(big)}")
        if not big.empty:
            print(big[["skater", "phase", "metric", "pct_change", "late_vs_early_effect"]].round(3).to_string(index=False))

    b, c = part_b_c_autoencoder(rows)
    if b is not None:
        print(f"\n{'='*78}\nB. AUTOENCODER reconstruction loss, early vs late (early-stats standardized)\n{'='*78}")
        print(b.round(4).to_string())
        print(f"\n{'='*78}\nC. CAMERA CONFOUND: does loss track on-screen body size?\n{'='*78}")
        print(c.round(3).to_string(index=False))
        print("  |rho| >= 0.3 means B's loss moves with camera zoom -- treat any B effect as")
        print("  confounded until the model is retrained on angle-only / rescaled features.")

    print("\nREADING THIS:")
    print("  - late_vs_early_effect: +1 = every late frame higher than every early frame,")
    print("    -1 = the reverse, 0 = no difference. Frames are autocorrelated, so no p-values.")
    print("  - Consistent direction across skaters matters more than any single number.")
    print("  - Assumption: late race = more fatigued. It is a proxy, not measured fatigue.")
    if not a.empty:
        print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
