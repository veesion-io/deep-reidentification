"""
Service orchestrator for running Steps 1 through 3 of the PoseTrack pipeline in an importable cloud-friendly manner.
"""
import sys
from pathlib import Path
import numpy as np

# Adjust paths manually if needed to import pipeline modules
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service.s3_downloader import download_from_s3
from pipeline.collate_videos import collate_all_cameras
from pipeline.prepare_data import prepare as prepare_data
from detection.get_detection_parallel import run_parallel_detection
from reid_parallel import run_parallel_reid
from track.run_tracking_batch import run_tracking_inference
from track.generate_submission import run_generate_submission


def run_pipeline(s3_uri: str, scene_name: str, base_dir: str = "/home/veesion/PoseTrack/online_sync", query_json: str = None):
    """
    Executes:
      1. Download S3 folder
      2. Collate video chunks into single videos
      3. Run the tracking pipeline
      4. (Optional) Find the user's query track ID
      5. (Optional) Find ReID matches across cameras
    """
    base_path = Path(base_dir)
    raw_s3_path = base_path / "raw" / scene_name
    collated_path = base_path / "videos" / scene_name

    print("\n" + "="*60)
    print(f"[Step 1] Downloading from S3: {s3_uri} -> {raw_s3_path}")
    print("="*60)
    download_from_s3(s3_uri, raw_s3_path)

    print("\n" + "="*60)
    print(f"[Step 2] Collating videos: {raw_s3_path} -> {collated_path}")
    print("="*60)
    collate_all_cameras(str(raw_s3_path), str(collated_path))

    print("\n" + "="*60)
    print("[Step 3] Running Pipeline Models")
    print("="*60)
    
    # 3.1 Convert collated videos into dataset structure
    dataset_dir = REPO_ROOT / "dataset" / "test" / scene_name
    print(f"\n=> Preparing Data into {dataset_dir}")
    prepare_data(videos_dir=collated_path, dataset_dir=dataset_dir)
    
    # 3.2 Detection
    print("\n=> Running Detection Parallel")
    # Tweak params for your standard setup
    run_parallel_detection(scene=scene_name, ckpt="yolo26m_people_detector.pt", conf=0.1, procs_per_gpu=6, devices="cuda:0,cuda:1", batch=16)

    # 3.3 ReID Feature Extraction
    print("\n=> Running ReID Feature Extraction")
    run_parallel_reid(scene=scene_name, procs_per_gpu=5, devices="cuda:0,cuda:1")

    # 3.4 Multi-camera Tracking
    print("\n=> Running Tracking Integration")
    result_dir = REPO_ROOT / "result"
    (result_dir / "track").mkdir(parents=True, exist_ok=True)
    (result_dir / "track_log").mkdir(parents=True, exist_ok=True)
    
    run_tracking_inference(scene_name=scene_name, no_reid_merge=False)

    # 3.5 Generate results
    print("\n=> Writing pipeline final output")
    run_generate_submission(scene=scene_name)

    track_file = result_dir / "track.txt"
    if track_file.exists():
        data = np.loadtxt(str(track_file))
        if data.size > 0:
            n_tracks = len(np.unique(data[:, 1]))
            n_dets = len(data)
            cams = np.unique(data[:, 0])
            print(f"\nResults: {n_tracks} people, {n_dets} detections across {len(cams)} cameras")
            print(f"Cameras: {cams}")
            print(f"Frame range: {data[:, 2].min():.0f} - {data[:, 2].max():.0f}")
        else:
            print("\nResult track.txt was empty.")
            
        if query_json:
            import os
            if not os.path.exists(query_json):
                print(f"WARNING: query JSON not found at {query_json}")
            else:
                from service.track_matcher import find_query_track, build_track_reid_features, find_reid_matches
                print("\n" + "="*60)
                print("[Step 4 & 5] Matching User Query Track via ReID")
                print("="*60)
                
                best_track_id, best_iou = find_query_track(query_json, str(track_file))
                if best_track_id == -1:
                    print("Failed to find query track overlapping with pipeline tracks!")
                else:
                    print(f"Found query track mapped to pipeline Track ID: {best_track_id} (IoU: {best_iou:.2f})")
                    
                    track_features = build_track_reid_features(str(track_file), scene_name, str(REPO_ROOT))
                    matches = find_reid_matches(best_track_id, track_features, threshold=0.5)
                    
                    print(f"Found {len(matches)} highly probable matches (sim > 0.5):")
                    for m in matches:
                        print(f"  Track ID: {m['track_id']} | Sim: {m['score']:.3f}")
                    
                    match_out_file = result_dir / f"{scene_name}_matches.json"
                    import json
                    with open(match_out_file, 'w') as f:
                        json.dump({"query_track_id": int(best_track_id), "matches": matches}, f, indent=2)
                    print(f"Matches saved to {match_out_file}")
                    
                    # --- Steps 6, 7, 8, 9 Extension ---
                    from service.video_generation import generate_pair_video, generate_timeline_video
                    from service.tier_service import send_for_annotation, send_for_alerting
                    
                    pairs_dir = result_dir / "pairs"
                    pairs_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate top N matches (e.g., top 5)
                    top_matches = matches[:5]
                    pair_video_paths = {}
                    
                    for i, m in enumerate(top_matches):
                        cand_id = m['track_id']
                        pair_path = str(pairs_dir / f"pair_{best_track_id}_vs_{cand_id}.mp4")
                        print(f"Generating pair video for candidate {cand_id} ({i+1}/{len(top_matches)})")
                        success = generate_pair_video(
                            query_track_id=best_track_id,
                            candidate_track_id=cand_id,
                            track_file_path=str(track_file),
                            videos_dir=str(collated_path),
                            output_path=pair_path
                        )
                        if success:
                            pair_video_paths[cand_id] = pair_path
                            
                    # Step 7: Send for annotation
                    if pair_video_paths:
                        approved_tracks = send_for_annotation(pair_video_paths)
                        
                        # Step 8: Build complete timeline
                        timeline_out = str(result_dir / f"timeline_{best_track_id}.mp4")
                        print(f"\n[Step 8] Building complete timeline for person {best_track_id}...")
                        generate_timeline_video(
                            query_track_id=best_track_id,
                            approved_track_ids=approved_tracks,
                            track_file_path=str(track_file),
                            videos_dir=str(collated_path),
                            output_path=timeline_out
                        )
                        
                        # Step 9: Alert client
                        print(f"\n[Step 9] Sending final timeline video to alerting service...")
                        send_for_alerting(timeline_out)
                    else:
                        print("No pair videos generated, skipping annotation, timeline, and alerting steps.")
                    # -------------------------------
    else:
        print("WARNING: result/track.txt not found!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Service Entrypoint")
    parser.add_argument("--s3-uri", type=str, required=True, help="S3 URI like s3://bucket/prefix/to/chunks")
    parser.add_argument("--scene", type=str, default="scene_cloud_001")
    parser.add_argument("--work-dir", type=str, default="/home/veesion/PoseTrack/online_sync")
    parser.add_argument("--query-json", type=str, default=None, help="Path to the JSON containing the user query track")
    args = parser.parse_args()

    run_pipeline(args.s3_uri, args.scene, args.work_dir, args.query_json)
