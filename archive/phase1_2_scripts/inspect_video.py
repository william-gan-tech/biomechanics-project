import cv2

video_path = "data/patrick_meek_3000m.mp4"
cap = cv2.VideoCapture(video_path)

# Get video metadata
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration_sec:.2f} seconds")

cap.release()