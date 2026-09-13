import pandas as pd
import cv2

df = pd.read_csv("ablation_feature_cache/Mia_Manganello_Kilburg_scaled_features.csv")

worst_idx = df["norm_right_shoulder_y"].abs().idxmax()
worst_frame = int(df.loc[worst_idx, "frame"])
worst_value = df.loc[worst_idx, "norm_right_shoulder_y"]
print(f"Most extreme norm_right_shoulder_y value: {worst_value:.3f} at frame {worst_frame} (row {worst_idx} of {len(df)})")
print(f"That is {100 * worst_idx / len(df):.1f}% through the sampled data")

diffs = df["norm_right_shoulder_y"].diff().abs()
biggest_jump_idx = diffs.idxmax()
biggest_jump_val = diffs.loc[biggest_jump_idx]
biggest_jump_frame = int(df.loc[biggest_jump_idx, "frame"])
print(f"\nBiggest single-frame jump: {biggest_jump_val:.3f} at frame {biggest_jump_frame}")

cap = cv2.VideoCapture("data/mia_manganello_kilburg.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, worst_frame)
ret, frame = cap.read()
if ret:
    cv2.imwrite("mia_worst_frame_check.jpg", frame)
    print("Saved mia_worst_frame_check.jpg -- open it to see what is happening at this point")
else:
    print("Could not read that frame")
cap.release()

# Also grab a frame near the biggest jump, in case it differs from the worst absolute value
cap2 = cv2.VideoCapture("data/mia_manganello_kilburg.mp4")
cap2.set(cv2.CAP_PROP_POS_FRAMES, biggest_jump_frame)
ret2, frame2 = cap2.read()
if ret2:
    cv2.imwrite("mia_jump_frame_check.jpg", frame2)
    print("Saved mia_jump_frame_check.jpg -- open it too, for comparison")
cap2.release()
