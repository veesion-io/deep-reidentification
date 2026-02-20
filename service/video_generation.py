"""
Video generation module for Steps 6 and 8 of the online pipeline.
Handles creating side-by-side comparison videos and chronological timeline videos.
"""
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from collections import defaultdict


def _get_tracks_by_id(track_file_path):
    """Parses track.txt and groups by track ID."""
    try:
        data = np.loadtxt(track_file_path)
    except Exception as e:
        print(f"Error loading track file {track_file_path}: {e}")
        return {}
        
    if data.ndim == 1:
        if len(data) == 0:
            return {}
        data = data[np.newaxis, :]
        
    tracks = defaultdict(list)
    # [cam, id, frame, x, y, w, h, xc, yc]
    for row in data:
        t_id = int(row[1])
        tracks[t_id].append({
            'cam': int(row[0]),
            'frame': int(row[2]),
            'bbox': [int(x) for x in row[3:7]]
        })
        
    for t_id in tracks:
        # Sort by frame
        tracks[t_id].sort(key=lambda x: x['frame'])
        
    return tracks

def _get_video_paths(videos_dir):
    """Finds video files mapping track camera ID to file path."""
    video_files = sorted(
        p for p in Path(videos_dir).iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    cam_videos = {}
    for idx, vp in enumerate(video_files, start=1):
        cam_videos[idx] = str(vp)
    return cam_videos

def _draw_box(frame, bbox, label, color=(0, 255, 0)):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(20, y - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def generate_pair_video(query_track_id, candidate_track_id, track_file_path, videos_dir, output_path, resolution=(640, 480)):
    """
    Generates a side-by-side video comparing the query track and the candidate track.
    (Step 6)
    """
    tracks = _get_tracks_by_id(track_file_path)
    if query_track_id not in tracks or candidate_track_id not in tracks:
        print(f"Missing track data for query ({query_track_id}) or candidate ({candidate_track_id})")
        return False
        
    q_track = tracks[query_track_id]
    c_track = tracks[candidate_track_id]
    
    q_frames = {det['frame']: det for det in q_track}
    c_frames = {det['frame']: det for det in c_track}
    
    min_frame = min(q_track[0]['frame'], c_track[0]['frame'])
    max_frame = max(q_track[-1]['frame'], c_track[-1]['frame'])
    
    cam_videos = _get_video_paths(videos_dir)
    
    # Needs to open videos as needed to save memory
    caps = {}
    
    # Output configuration
    out_w, out_h = resolution[0] * 2, resolution[1]  # Side by side
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, 25.0, (out_w, out_h))
    
    if not writer.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return False

    print(f"Generating pair video {output_path} (frames {min_frame} - {max_frame})")
    
    # Prepare blank frame
    blank_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    
    for f in range(min_frame, max_frame + 1):
        # Left side: Query
        left_frame = blank_frame.copy()
        if f in q_frames:
            det = q_frames[f]
            cam_idx = det['cam']
            if cam_idx in cam_videos:
                if cam_idx not in caps:
                    caps[cam_idx] = cv2.VideoCapture(cam_videos[cam_idx])
                
                cap = caps[cam_idx]
                cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)  # 0-indexed vs 1-indexed detection
                ret, frame = cap.read()
                if ret:
                    # Draw before resize to use original bbox coordinates
                    _draw_box(frame, det['bbox'], f"Query ID: {query_track_id}", (0, 255, 0))
                    left_frame = cv2.resize(frame, resolution)
                    cv2.putText(left_frame, f"Cam {cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        # Right side: Candidate
        right_frame = blank_frame.copy()
        if f in c_frames:
            det = c_frames[f]
            cam_idx = det['cam']
            if cam_idx in cam_videos:
                if cam_idx not in caps:
                    caps[cam_idx] = cv2.VideoCapture(cam_videos[cam_idx])
                
                cap = caps[cam_idx]
                cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
                ret, frame = cap.read()
                if ret:
                    _draw_box(frame, det['bbox'], f"Cand ID: {candidate_track_id}", (0, 165, 255)) # Orange
                    right_frame = cv2.resize(frame, resolution)
                    cv2.putText(right_frame, f"Cam {cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        # Concat and write
        pair_frame = np.concatenate((left_frame, right_frame), axis=1)
        writer.write(pair_frame)
        
    writer.release()
    for cap in caps.values():
        cap.release()
        
    return True


def generate_timeline_video(query_track_id, approved_track_ids, track_file_path, videos_dir, output_path, cell_res=(640, 480)):
    """
    Generates the chronological timeline video for the person.
    (Step 8)
    """
    tracks = _get_tracks_by_id(track_file_path)
    
    timeline_tracks = {query_track_id: tracks.get(query_track_id, [])}
    for t_id in approved_track_ids:
        if t_id in tracks:
            timeline_tracks[t_id] = tracks[t_id]
            
    # Map frame -> list of detections across cameras
    frame_to_dets = defaultdict(list)
    for t_id, dets in timeline_tracks.items():
        for det in dets:
            det_copy = det.copy()
            det_copy['track_id'] = t_id
            frame_to_dets[det['frame']].append(det_copy)
            
    if not frame_to_dets:
        print("No frames to generate timeline from.")
        return False
        
    min_frame = min(frame_to_dets.keys())
    max_frame = max(frame_to_dets.keys())
    
    cam_videos = _get_video_paths(videos_dir)
    caps = {}
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We dynamically create/recreate writers depending on the number of active cameras,
    # OR we make a consistent large output and place frames inside it.
    # The requirement: "jumps from 09 to 11 with only camera 2, then at 12, camera 3 feed appears on the right"
    # It indicates dynamic width is preferred or at least side-by-side if multiple.
    # To use a standard VideoWriter, width/height must remain constant.
    # To support dynamic width, we can set the width to maximum required (e.g., 2 or 3 cameras max overlapping).
    
    # Find max overlapping cameras in any single frame
    max_overlap = max((len(dets) for dets in frame_to_dets.values()), default=1)
    
    out_w, out_h = cell_res[0] * max_overlap, cell_res[1]
    writer = cv2.VideoWriter(output_path, fourcc, 25.0, (out_w, out_h))
    
    if not writer.isOpened():
        print(f"Failed to open timeline video writer for {output_path}")
        return False
        
    print(f"Generating timeline video {output_path} (frames {min_frame} - {max_frame}, max overlap {max_overlap})")
    blank_cell = np.zeros((cell_res[1], cell_res[0], 3), dtype=np.uint8)
    
    for f in range(min_frame, max_frame + 1):
        dets = frame_to_dets.get(f, [])
        if not dets:
            # Skip frames where person is not visible
            continue
            
        cells = []
        for det in dets:
            cam_idx = det['cam']
            if cam_idx in cam_videos:
                if cam_idx not in caps:
                    caps[cam_idx] = cv2.VideoCapture(cam_videos[cam_idx])
                
                cap = caps[cam_idx]
                cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
                ret, frame = cap.read()
                if ret:
                    _draw_box(frame, det['bbox'], f"ID: {det['track_id']}", (0, 255, 255)) # Yellow
                    cell = cv2.resize(frame, cell_res)
                    cv2.putText(cell, f"Cam {cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cells.append(cell)
                    
        # Pad with black cells to reach out_w
        while len(cells) < max_overlap:
            cells.append(blank_cell.copy())
            
        if cells:
            row_frame = np.concatenate(cells, axis=1)
            writer.write(row_frame)
            
    writer.release()
    for cap in caps.values():
        cap.release()
        
    print(f"Timeline video saved to {output_path}")
    return True
