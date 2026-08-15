import os
from yt_dlp import YoutubeDL

# Define output path
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
output_path = os.path.join(base_dir, 'data', 'sven_kramer_ref.mp4')

url = 'https://www.youtube.com/watch?v=Vdk03UWwd30'

ydl_opts = {
    'format': 'best[ext=mp4]/best', 
    'outtmpl': output_path,
    # ADD THESE LINES TO BYPASS THE 403 ERROR:
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'noplaylist': True,
}

with YoutubeDL(ydl_opts) as ydl:
    print(f"Downloading elite reference video from: {url}")
    ydl.download([url])
    print(f"Successfully saved to: {output_path}")