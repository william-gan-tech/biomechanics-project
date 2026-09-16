import os
from moviepy import VideoFileClip, AudioFileClip

os.makedirs("outputs/meek_segments", exist_ok=True)

def process_clip(video_path, audio_path, start, end, output_path):
    video = VideoFileClip(video_path).subclipped(start, end)
    audio = AudioFileClip(audio_path).subclipped(start, end)
    final = video.with_audio(audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

video_file = "data/full_video.f137.mp4"
audio_file = "data/full_video.f251.webm"

process_clip(video_file, audio_file, 55, 85, "outputs/meek_segments/segment_1.mp4")
process_clip(video_file, audio_file, 255, 285, "outputs/meek_segments/segment_2.mp4")