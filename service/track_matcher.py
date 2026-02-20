import numpy as np
import json
import os.path as osp
import glob

def compute_iou(box1, box2):
    # box: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def find_query_track(query_json_path, track_file_path):
    with open(query_json_path, 'r') as f:
        query_data = json.load(f)
    
    cam_str = query_data['camera']
    query_track = query_data['track'] # list of {"frame_id": int, "bbox": [x, y, w, h]}
    
    cam_idx = int(cam_str.split('_')[-1])
    
    # Load tracks: [cam, id, frame, x, y, w, h, xc, yc]
    track_data = np.loadtxt(track_file_path)
    
    if track_data.ndim == 1:
        if len(track_data) == 0:
            return -1, 0
        track_data = track_data[np.newaxis, :]
        
    cam_tracks = track_data[track_data[:, 0] == cam_idx]
    
    # Group by track_id
    track_ious = {}
    
    for q in query_track:
        q_frame = q['frame_id']
        q_box = q['bbox']
        
        # Find all tracks in this frame
        frame_tracks = cam_tracks[cam_tracks[:, 2] == q_frame]
        for t in frame_tracks:
            t_id = int(t[1])
            t_box = t[3:7]
            iou = compute_iou(q_box, t_box)
            if t_id not in track_ious:
                track_ious[t_id] = []
            track_ious[t_id].append(iou)
            
    # Calculate average IOU for each track_id considering all queried frames
    best_track_id = -1
    best_iou = -1
    for t_id, ious in track_ious.items():
        avg_iou = sum(ious) / len(query_track)  # divide by query length to penalize missing frames
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_track_id = t_id
            
    return best_track_id, best_iou

def load_reid_features(scene_name, base_dir):
    reid_dir = osp.join(base_dir, "result/reid", scene_name)
    det_dir = osp.join(base_dir, "result/detection", scene_name)
    
    features = {}
    # features[cam_idx] = {"frames": [], "bboxes": [], "features": []}
    
    if not osp.exists(reid_dir):
        return features
        
    for reid_file in glob.glob(osp.join(reid_dir, "*.npy")):
        cam_str = osp.basename(reid_file).split('.')[0] # e.g. camera_0001
        cam_idx = int(cam_str.split('_')[-1])
        
        det_file = osp.join(det_dir, f"{cam_str}.txt")
        if not osp.exists(det_file):
            continue
            
        det_data = np.loadtxt(det_file, delimiter=',')
        if det_data.ndim == 1:
            if len(det_data) == 0:
                continue
            det_data = det_data[np.newaxis, :]
            
        reid_data = np.load(reid_file)
        
        features[cam_idx] = {
            "frames": det_data[:, 0],
            "bboxes": det_data[:, 2:6], # x, y, w, h
            "features": reid_data
        }
        
    return features

def build_track_reid_features(track_file_path, scene_name, base_dir):
    track_data = np.loadtxt(track_file_path)
    if track_data.ndim == 1:
        if len(track_data) == 0:
            return {}
        track_data = track_data[np.newaxis, :]
        
    reid_data = load_reid_features(scene_name, base_dir)
    
    # Compile features for each track_id
    track_features = {}
    
    for row in track_data:
        cam_idx = int(row[0])
        t_id = int(row[1])
        frame = row[2]
        box = row[3:7]
        
        if cam_idx not in reid_data:
            continue
            
        cam_reid = reid_data[cam_idx]
        
        # Find matching detection
        frame_mask = cam_reid["frames"] == frame
        if not np.any(frame_mask):
            continue
            
        frame_bboxes = cam_reid["bboxes"][frame_mask]
        frame_feats = cam_reid["features"][frame_mask]
        
        best_iou = 0
        best_feat = None
        for i, fbox in enumerate(frame_bboxes):
            iou = compute_iou(box, fbox)
            if iou > best_iou:
                best_iou = iou
                best_feat = frame_feats[i]
                
        if best_iou > 0.5 and best_feat is not None:
            if t_id not in track_features:
                track_features[t_id] = []
            track_features[t_id].append(best_feat)
            
    # Average features
    track_avg_features = {}
    for t_id, feats in track_features.items():
        avg_feat = np.mean(feats, axis=0)
        # L2 normalize
        avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-12)
        track_avg_features[t_id] = avg_feat
        
    return track_avg_features

def find_reid_matches(query_track_id, track_features, threshold=0.5):
    if query_track_id not in track_features:
        return []
        
    query_feat = track_features[query_track_id]
    
    matches = []
    for t_id, feat in track_features.items():
        if t_id == query_track_id:
            continue
        sim = np.dot(query_feat, feat)
        if sim > threshold:
            matches.append({"track_id": t_id, "score": float(sim)})
            
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches
