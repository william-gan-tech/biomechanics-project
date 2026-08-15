import os
import cv2

# Dynamically resolve the project root directory based on the location of this script
current_dir = os.path.dirname(os.path.abspath(__file__)) # points to 'src/'
base_dir = os.path.dirname(current_dir) # points to 'biomechanics-project/'

video_path = os.path.join(base_dir, 'data', 'sven_kramer_ref.mp4')

print(f"Target video path: {video_path}")

# Verify that the file was successfully placed in the data folder
if os.path.exists(video_path):
    print("Success! The reference video file was found.")
    
    # Open the video with OpenCV to verify it reads frames properly
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    cap.release()
    
else:
    print("Error: Could not find 'sven_kramer_ref.mp4' in the data folder.")