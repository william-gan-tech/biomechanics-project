import cv2

# Open your main full video (replace with your original video filename if needed)
cap = cv2.VideoCapture("data/skater.mp4") # or whatever your source video is named

fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(225 * fps)  # 3 minutes 45 seconds
end_frame = int(254 * fps)    # 4 minutes 14 seconds

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("data/trimmed_skater_end.mp4", fourcc, fps, (width, height))

current_frame = start_frame
while current_frame <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    current_frame += 1

cap.release()
out.release()
print("✅ Successfully created trimmed_skater_end.mp4!")