import json
import os
import cv2
import numpy as np

def convert_scene(scene_id="scene_001"):
    pose_dir = f"result/pose/{scene_id}"
    track_file = "result/track.txt"
    det_output_dir = f"CasCalib/example_data/detections"
    frame_output_dir = f"CasCalib/example_data/frames"
    os.makedirs(det_output_dir, exist_ok=True)
    os.makedirs(frame_output_dir, exist_ok=True)

    # Load tracks
    tracks = {} # (cam_id, frame_id) -> list of (track_id, bbox)
    if os.path.exists(track_file):
        with open(track_file, "r") as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                cam_id = int(parts[0])
                track_id = int(parts[1])
                frame_id = int(parts[2])
                bbox = [float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])] # x y w h
                key = (cam_id, frame_id)
                if key not in tracks:
                    tracks[key] = []
                tracks[key].append((track_id, bbox))

    # Process each camera pose file
    cam_names = []
    for filename in sorted(os.listdir(pose_dir)):
        if not filename.startswith("camera_") or not filename.endswith(".txt"):
            continue
        
        cam_num = int(filename.split("_")[1].split(".")[0])
        input_path = os.path.join(pose_dir, filename)
        
        # Determine camera name for CasCalib
        cam_name = f"{scene_id}-c{cam_num-1}"
        cam_names.append(cam_name)
        
        info_list = []
        with open(input_path, "r") as f:
            for line in f:
                parts = [float(p) for p in line.split()]
                if not parts: continue
                
                frame_id = int(parts[0])
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = parts[1:5]
                score = parts[5]
                keypoints_flat = parts[6:]
                keypoints = []
                for i in range(0, len(keypoints_flat), 3):
                    keypoints.append([keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]])
                
                w = bbox_x2 - bbox_x1
                h = bbox_y2 - bbox_y1
                
                # Match track_id
                track_id = -1
                # Note: PoseTrack camera files are numbered 1-based. 
                # track.txt might use 1-based or 0-based cam_id.
                # Based on previous check, track.txt has cam_ids 2, 4, 5...
                # Which matches camera_0002.txt, camera_0004.txt...
                key = (cam_num, frame_id)
                if key in tracks:
                    best_match = -1
                    min_dist = 1000
                    for tid, tbbox in tracks[key]:
                        dist = abs(bbox_x1 - tbbox[0]) + abs(bbox_y1 - tbbox[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_match = tid
                    if min_dist < 50: # Slightly relaxed threshold
                        track_id = best_match
                
                if track_id == -1:
                    # Generic ID if not tracked
                    track_id = (cam_num * 100000) + frame_id 
                
                info_list.append({
                    "bbox": [bbox_x1, bbox_y1, w, h, score],
                    "keypoints": keypoints,
                    "area": w * h,
                    "track_id": track_id,
                    "frame": frame_id
                })
        
        # Save to JSON for CasCalib
        output_name = f"result_{cam_name}_.json"
        det_output_path = os.path.join(det_output_dir, output_name)
        with open(det_output_path, "w") as f:
            json.dump({"Info": info_list}, f)
        print(f"Written detections: {det_output_path}")

        # Create dummy frame for resolution
        video_path = f"dataset/test/{scene_id}/camera_{cam_num:04d}/video.mp4"
        frame_cam_dir = os.path.join(frame_output_dir, f"{cam_name}_avi")
        os.makedirs(frame_cam_dir, exist_ok=True)
        dummy_frame_path = os.path.join(frame_cam_dir, "00000000.jpg")
        
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(dummy_frame_path, frame)
                print(f"Written frame: {dummy_frame_path}")
            cap.release()
        else:
            # Fallback dummy image 2112x2112
            dummy_img = np.zeros((2112, 2112, 3), dtype=np.uint8)
            cv2.imwrite(dummy_frame_path, dummy_img)
            print(f"Written fallback frame: {dummy_frame_path}")

    return cam_names

if __name__ == "__main__":
    convert_scene()
