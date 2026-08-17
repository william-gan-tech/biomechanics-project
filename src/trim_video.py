from moviepy import VideoFileClip

# Replace "skater_time_trial.mp4" with your actual downloaded video filename
input_filename = "skater_time_trial.mp4"
output_filename = "data/trimmed_skater.mp4"

try:
    # Load your downloaded video
    clip = VideoFileClip(input_filename)
    
    # Choose your start and end times in seconds (e.g., from 60 seconds to 90 seconds)
    start_time = 20
    end_time = 50
    
    print(f"Trimming video from {start_time}s to {end_time}s...")
    trimmed_clip = clip.subclipped(start_time, end_time)
    
    # Save the trimmed clip
    trimmed_clip.write_videofile(output_filename, fps=30)
    print(f"✅ Success! Trimmed video saved to '{output_filename}'")

except Exception as e:
    print(f"❌ Error trimming video: {e}")