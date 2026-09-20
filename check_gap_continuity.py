"""
Decisive same-skater check: instead of relying only on appearance
similarity (which came back ambiguous at 0.6964), this checks whether the
POSITION right before the data gap and right after it is physically
consistent with ONE continuously accelerating skater -- or whether it
implies an impossible instantaneous jump, which would mean a different
skater was picked up after the gap.

Usage:
    python -m check_gap_continuity
"""

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

VIDEO_NAME = "start_candidate_3"
VIDEO_PATH = "data/start_candidate_3.mp4"

LAST_REST_FRAME = 169
FIRST_ACCEL_FRAME = 193
GAP_FRAMES = FIRST_ACCEL_FRAME - LAST_REST_FRAME


def main():
    df = get_skater_features(VIDEO_NAME, VIDEO_PATH, "scaled")
    if df is None:
        print("Could not load features.")
        return
    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    last_rest = df[df["frame"] == LAST_REST_FRAME]
    first_accel = df[df["frame"] == FIRST_ACCEL_FRAME]

    if last_rest.empty or first_accel.empty:
        print(f"Missing data at frame {LAST_REST_FRAME} or {FIRST_ACCEL_FRAME} -- adjust frame numbers.")
        return

    rest_x = last_rest["norm_right_hip_x"].values[0]
    rest_y = last_rest["norm_right_hip_y"].values[0]
    accel_x = first_accel["norm_right_hip_x"].values[0]
    accel_y = first_accel["norm_right_hip_y"].values[0]

    jump_distance = ((accel_x - rest_x) ** 2 + (accel_y - rest_y) ** 2) ** 0.5
    gap_seconds = GAP_FRAMES / 30.0
    implied_velocity = jump_distance / gap_seconds

    print(f"Position at end of REST (frame {LAST_REST_FRAME}): ({rest_x:.4f}, {rest_y:.4f})")
    print(f"Position at start of EARLY-ACCELERATION (frame {FIRST_ACCEL_FRAME}): ({accel_x:.4f}, {accel_y:.4f})")
    print(f"Distance moved across the {GAP_FRAMES}-frame ({gap_seconds:.2f}s) gap: {jump_distance:.4f}")
    print(f"Implied average velocity during the gap: {implied_velocity:.4f} (normalized units/sec)\n")

    accel_segment = df[(df["frame"] >= FIRST_ACCEL_FRAME) & (df["frame"] <= FIRST_ACCEL_FRAME + 50)]
    observed_max_velocity = accel_segment["hip_velocity"].max() * 30.0

    print(f"Max velocity actually observed in the 50 frames after the gap: {observed_max_velocity:.4f} (per-second equivalent)\n")

    if implied_velocity <= observed_max_velocity * 1.5:
        print("CONSISTENT: the implied velocity across the gap is in the same ballpark as")
        print("velocities actually observed right after -- physically plausible for ONE")
        print("continuously accelerating skater. This supports 'same skater' more decisively")
        print("than the appearance check alone.")
    else:
        print("INCONSISTENT: the implied velocity across the gap is much higher than anything")
        print("observed afterward -- this would require an implausible instantaneous jump,")
        print("suggesting a DIFFERENT skater was picked up after the gap.")


if __name__ == "__main__":
    main()
