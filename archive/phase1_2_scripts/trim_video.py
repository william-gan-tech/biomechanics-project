import cv2
import os

def trim_fatigued_video():
    # Make sure data folder exists
    os.makedirs("data", exist_ok=True)
    
    # Path to your original source video
    source_video = "data/sven_kramer_ref.mp4"
    output_video = "data/trimmed_skater_end.mp4"
    
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        print(f"Error: Could not open source video file at {source_video}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0 # Default fallback if FPS can't be read

    # 3 minutes 45 seconds to 4 minutes 14 seconds
    start_frame = int(225 * fps)  
    end_frame = int(254 * fps)    

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use mp4v codec for proper container writing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_video}.")
        cap.release()
        return

    print(f"Trimming frames from {start_frame} to {end_frame}...")
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

if __name__ == "__main__":
    trim_fatigued_video()