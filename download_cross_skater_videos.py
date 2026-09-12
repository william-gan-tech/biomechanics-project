"""
Batch-downloads video for skaters that don't have local footage yet, so they
can be added to CROSS_SKATER_VIDEO_MAP for real (not CSV-fallback or
simulated) cross-skater comparisons.

Usage:
    python -m download_cross_skater_videos

Each entry below downloads to data/<skater_slug>.mp4. After running, add
the printed paths into CROSS_SKATER_VIDEO_MAP in app.py.
"""

import os
from pipeline_engine import download_video_from_url

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SKATER_URLS = {
    "Ragne Wiklund": "https://www.youtube.com/watch?v=LyQQ23xfj4s",
    "Mia Manganello Kilburg": "https://www.youtube.com/watch?v=IeneKwWy25Q&t=1s",
    "Jorrit Bergsma": "https://www.youtube.com/watch?v=mi9bcc_w-Tw",
    "Jan Blokhuijsen": "https://www.youtube.com/watch?v=G2ixTdZmIhY",
    # Sandrina Tas intentionally omitted -- no video found.
}


def slugify(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def main():
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    results = {}

    for skater_name, url in SKATER_URLS.items():
        slug = slugify(skater_name)
        output_path = os.path.join(ROOT_DIR, "data", f"{slug}.mp4")

        print(f"\nDownloading {skater_name} -> {output_path}")
        success, result = download_video_from_url(url, output_path=output_path)

        if success:
            size_mb = os.path.getsize(result) / (1024 * 1024)
            print(f"  OK ({size_mb:.1f} MB)")
            results[skater_name] = f"data/{slug}.mp4"
        else:
            print(f"  FAILED: {result}")
            results[skater_name] = None

    print("\n" + "=" * 60)
    print("Add these entries to CROSS_SKATER_VIDEO_MAP in app.py:")
    print("=" * 60)
    for skater_name, path in results.items():
        if path:
            print(f'    "{skater_name}": "{path}",')
        else:
            print(f'    "{skater_name}": None,  # download failed, see log above')


if __name__ == "__main__":
    main()
