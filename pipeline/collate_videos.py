import os
import subprocess
import json
import argparse
from pathlib import Path
import shutil
import statistics

DEFAULT_SOURCE_DIR = "/home/veesion/hq_cameras/es-min-33011-mieres-6"
DEST_DIR = "/home/veesion/PoseTrack/videos"
MIN_FILE_SIZE = 1000  # bytes, discard smaller chunks
SYNC_THRESHOLD_SEC = 300  # 5 minutes

def get_chunk_id(filename):
    try:
        return int(filename.split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        return -1

def collate_camera(cam_dir):
    cam_name = cam_dir.name
    chunks = sorted([f for f in cam_dir.glob("chunk_*.mp4") if f.stat().st_size > MIN_FILE_SIZE], key=lambda x: get_chunk_id(x.name))
    
    if not chunks:
        print(f"No valid chunks found for {cam_name}")
        return None, None

    # Get start time from the first valid chunk
    start_time = chunks[0].stat().st_mtime
    
    # Create concat file for ffmpeg
    concat_list_path = cam_dir / "concat_list.txt"
    with open(concat_list_path, 'w') as f:
        for chunk in chunks:
            f.write(f"file '{chunk.name}'\n")
    
    temp_output = cam_dir / f"{cam_name}.mp4"
    if temp_output.exists():
        temp_output.unlink()

    print(f"Collating {len(chunks)} chunks for {cam_name}...")
    try:
        # Use ffmpeg to concatenate without re-encoding
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
            '-i', str(concat_list_path), '-c', 'copy', str(temp_output)
        ], check=True, capture_output=True)
        
        # Preserve creation/modify time
        os.utime(temp_output, (start_time, start_time))
        
        return temp_output, start_time
    except subprocess.CalledProcessError as e:
        print(f"Error collating {cam_name}: {e.stderr.decode()}")
        return None, None
    finally:
        if concat_list_path.exists():
            concat_list_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="Collate video chunks into single camera videos.")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE_DIR, help="Source directory containing camera folders.")
    parser.add_argument("--dest", type=str, default=DEST_DIR, help="Destination directory for collated videos.")
    args = parser.parse_args()

    source_path = Path(args.source)
    dest_path = Path(args.dest)
    
    # Clear destination directory (optional but recommended in plan)
    if dest_path.exists():
        print(f"Clearing existing videos in {DEST_DIR}...")
        for f in dest_path.glob("*.mp4"):
            f.unlink()
    else:
        dest_path.mkdir(parents=True, exist_ok=True)

    cameras = [d for d in source_path.iterdir() if d.is_dir()]
    results = []

    for cam_dir in cameras:
        output_file, start_time = collate_camera(cam_dir)
        if output_file:
            results.append({
                "path": output_file,
                "start_time": start_time,
                "name": cam_dir.name
            })

    if not results:
        print("No videos were collated.")
        return

    # Sync Filtering
    start_times = [r["start_time"] for r in results]
    median_start = statistics.median(start_times)
    
    print(f"Median start time: {median_start}")
    
    synced_results = []
    for r in results:
        diff = abs(r["start_time"] - median_start)
        if diff > SYNC_THRESHOLD_SEC:
            print(f"Discarding {r['name']} - out of sync by {diff:.1f}s")
            r["path"].unlink()
        else:
            synced_results.append(r)

    # Move to final destination
    for r in synced_results:
        final_path = dest_path / f"{r['name']}.mp4"
        shutil.move(str(r["path"]), str(final_path))
        print(f"Finalized: {final_path.name}")

if __name__ == "__main__":
    main()
