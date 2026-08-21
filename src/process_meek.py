import cv2
import os

def trim_video_segments(input_path="data/patrick_meek_3000m.mp4", output_dir="outputs/meek_segments/"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Updated frame boundaries based on your target timestamps
    segments = {
        "fresh_state": (899, 1798),     # 30s to 60s
        "fatigued_state": (2997, 3896)  # 1:40 to 2:10
    }
    
    for segment_name, (start_frame, end_frame) in segments.items():
        output_path = os.path.join(output_dir, f"{segment_name}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        while current_frame <= end_frame and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1
            
        out.release()
        print(f"Successfully saved: {output_path} (Frames {start_frame} to {end_frame})")
        
    cap.release()

if __name__ == "__main__":
    trim_video_segments()