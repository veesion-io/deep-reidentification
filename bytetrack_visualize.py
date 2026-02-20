#!/usr/bin/env python3
"""
Simple ByteTrack visualization: collate camera chunks, run YOLO + ByteTrack,
and export annotated videos with bounding boxes and track IDs.

Usage:
    python3 bytetrack_visualize.py --source /home/veesion/hq_cameras/au-iga-4169-lytton-33
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ── Defaults ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = str(REPO_ROOT / "yolo26m_people_detector.pt")
MIN_CHUNK_SIZE = 1000  # bytes – skip tiny/corrupt chunks

# Distinct colours for track IDs
_PALETTE = [
    (255, 100, 50), (50, 255, 100), (50, 100, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (200, 128, 0), (0, 200, 128),
    (128, 0, 200), (200, 200, 200), (100, 180, 255), (255, 180, 100),
]


def _color(tid: int) -> tuple:
    return _PALETTE[tid % len(_PALETTE)]


# ── Collate chunked mp4s into one video ──────────────────────────────────────
def collate_camera(cam_dir: Path, out_dir: Path) -> Path | None:
    """Concatenate chunk_*.mp4 files in *cam_dir* into a single mp4 in *out_dir*."""
    chunks = sorted(
        [f for f in cam_dir.glob("chunk_*.mp4") if f.stat().st_size > MIN_CHUNK_SIZE],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not chunks:
        print(f"  ⚠  No valid chunks in {cam_dir.name}")
        return None

    concat_list = cam_dir / "_concat.txt"
    try:
        with open(concat_list, "w") as f:
            for c in chunks:
                f.write(f"file '{c.name}'\n")

        out_path = out_dir / f"{cam_dir.name}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-c", "copy", str(out_path),
            ],
            check=True, capture_output=True,
        )
        return out_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗  ffmpeg failed for {cam_dir.name}: {e.stderr.decode()[:200]}")
        return None
    finally:
        concat_list.unlink(missing_ok=True)


# ── Draw tracking results on a frame ─────────────────────────────────────────
def draw_tracks(frame: np.ndarray, boxes, track_ids, confs) -> np.ndarray:
    """Draw bounding boxes and track IDs on *frame* (in-place)."""
    for box, tid, conf in zip(boxes, track_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        col = _color(int(tid))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"ID {int(tid)}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


# ── Process one camera video ─────────────────────────────────────────────────
def process_video(model: YOLO, video_path: Path, out_path: Path,
                  device: str, conf: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗  Cannot open {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    # ffmpeg concat often produces insane fps (1000+); clamp to sane range
    fps = raw_fps if raw_fps and 1.0 <= raw_fps <= 60.0 else 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"  ⚠  mp4v writer failed (fps={raw_fps}), trying XVID→avi fallback")
        out_path = out_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Use model.track() which streams frames internally
    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        device=device,
        conf=conf,
        iou=0.5,
        stream=True,        # yields frame-by-frame, saves memory
        verbose=False,
        persist=True,        # keep tracks across frames
    )

    for r in tqdm(results, total=total, desc=f"  {video_path.stem}", unit="fr"):
        frame = r.orig_img
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            draw_tracks(frame, boxes, ids, confs)
        writer.write(frame)

    writer.release()
    print(f"  ✓  Saved {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO + ByteTrack on all cameras and export annotated videos."
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Store directory, e.g. /home/veesion/hq_cameras/au-iga-4169-lytton-33",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: result/<store>/bytetrack_videos/)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    store_name = source.name
    cameras = sorted([d for d in source.iterdir() if d.is_dir()])
    print(f"Store: {store_name}  —  {len(cameras)} cameras found")

    # ── Collate videos into a separate temp-style directory ──────────────
    collated_dir = REPO_ROOT / "collated_tmp" / store_name
    collated_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.output) if args.output else REPO_ROOT / "result" / store_name / "bytetrack_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Collating camera chunks into {collated_dir} ──")
    collated_videos: list[Path] = []
    for cam in cameras:
        existing = collated_dir / f"{cam.name}.mp4"
        if existing.exists():
            print(f"  ↻  {cam.name} already collated, reusing")
            collated_videos.append(existing)
            continue
        vid = collate_camera(cam, collated_dir)
        if vid:
            collated_videos.append(vid)

    if not collated_videos:
        print("No videos to process!")
        return

    # ── Run detection + ByteTrack on each video ──────────────────────────
    print(f"\n── Running YOLO + ByteTrack (conf={args.conf}) ──")
    model = YOLO(args.model)

    for video in collated_videos:
        out_file = out_dir / video.name
        if out_file.exists() and out_file.stat().st_size > 1000:
            print(f"\n↻  {video.name} already processed, skipping")
            continue
        print(f"\nProcessing {video.name}:")
        process_video(model, video, out_file, args.device, args.conf)
        # Reset tracker state between cameras
        model.predictor = None

    print(f"\n{'='*60}")
    print(f"Done! {len(collated_videos)} annotated videos saved to:")
    print(f"  {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
