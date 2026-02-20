"""
Mock service for Step 7 of the online pipeline.
Simulates sending video pairs to a tier service for human annotation and awaiting the response.
"""
import time
import random


def send_for_annotation(pair_videos: dict) -> list[int]:
    """
    Simulates sending video pairs for human annotation.
    
    Args:
        pair_videos: A dictionary mapping candidate track IDs to their pair video paths.
                     Example: {candidate_track_id: "/path/to/pair_video.mp4"}
                     
    Returns:
        A list of candidate track IDs that were approved.
    """
    print(f"\n[Tier Service] Received {len(pair_videos)} pair videos for annotation.")
    
    # Mocking network delay/human annotation time
    print("[Tier Service] Awaiting human annotation...")
    time.sleep(2.0)
    
    approved_tracks = []
    
    for track_id, video_path in pair_videos.items():
        # For the mock, we simulate that human annotators approve most highly confident matches,
        # but for testing purposes we can just accept all or randomly accept.
        # Here we will just accept all to ensure Step 8 has data to work with.
        print(f"  -> Annotator reviewed {video_path}")
        print(f"  -> Match with track {track_id} APPROVED.")
        approved_tracks.append(track_id)
        
    print(f"[Tier Service] Annotation complete. {len(approved_tracks)} matches approved.\n")
    return approved_tracks


def send_for_alerting(timeline_video_path: str) -> bool:
    """
    Simulates sending the final complete timeline video to a tier service
    for client alerting.
    
    Args:
        timeline_video_path: Path to the final timeline video of the query track.
        
    Returns:
        True if the alerting service successfully received it.
    """
    print(f"\n[Alert Service] Received final timeline video for alerting: {timeline_video_path}")
    
    # Mocking network delay
    print("[Alert Service] Alerting client...")
    time.sleep(1.0)
    
    print("[Alert Service] Client successfully alerted.\n")
    return True

