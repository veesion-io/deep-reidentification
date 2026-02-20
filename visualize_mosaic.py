#!/usr/bin/env python3
"""Generate a mosaic video of all cameras with tracking bounding boxes and person IDs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _probe_fps(video_path: Path) -> float | None:
    """Get real FPS from the codec-level r_frame_rate reported by ffprobe.

    Container durations are often wrong after concat-copy, so we read the
    stream's r_frame_rate (the codec tick rate) which reflects the true
    capture rate of the camera.  Returns None for insane values (>100).
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video_path)],
            stderr=subprocess.DEVNULL,
        )
        for stream in json.loads(out).get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            rfr = stream.get("r_frame_rate", "")
            if "/" in rfr:
                num, den = rfr.split("/")
                fps = int(num) / int(den)
            else:
                fps = float(rfr)
            if 1 < fps < 100:
                return fps
    except Exception:
        pass
    return None

# ── Configuration ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "dataset" / "test" / "scene_001"
TRACK_FILE = REPO_ROOT / "result" / "track.txt"
OUTPUT_FILE = REPO_ROOT / "result" / "mosaic_tracks.mp4"

NUM_COLS = 4
CELL_W = 480
CELL_H = 384
OUTPUT_FPS = None  # Auto-detect from source videos (fallback: 25)

# High-contrast, distinguishable colours per person ID  (BGR)
PERSON_COLORS = {
    1: (0, 200, 255),    # orange
    2: (255, 50, 50),    # blue
    3: (50, 255, 50),    # green
    4: (50, 50, 255),    # red
    5: (255, 255, 0),    # cyan
    6: (180, 0, 255),    # magenta
    7: (0, 255, 255),    # yellow
    8: (255, 150, 0),    # teal
    9: (100, 200, 255),  # light orange
}


def _fallback_color(pid: int) -> tuple[int, int, int]:
    """Generate a deterministic colour for any person ID not in the palette."""
    rng = np.random.RandomState(pid * 37 + 7)
    return tuple(int(c) for c in rng.randint(80, 255, size=3))


def color_for(pid: int) -> tuple[int, int, int]:
    return PERSON_COLORS.get(pid) or _fallback_color(pid)


# ── Load tracking data ───────────────────────────────────────────────────────
def load_tracks(path: Path) -> dict[tuple[int, int], list[tuple[int, int, int, int, int]]]:
    """Return {(cam_id, frame_id): [(person_id, x, y, w, h), ...]}."""
    tracks: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            cam = int(float(parts[0]))
            pid = int(float(parts[1]))
            fid = int(float(parts[2]))
            x, y, w, h = int(float(parts[3])), int(float(parts[4])), int(float(parts[5])), int(float(parts[6]))
            tracks[(cam, fid)].append((pid, x, y, w, h))
    return dict(tracks)


# ── Draw helpers ─────────────────────────────────────────────────────────────
def draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple, font_scale: float = 0.7, thickness: int = 2):
    """Draw text with a dark shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_tracks_on_frame(frame: np.ndarray, detections: list, orig_w: int, orig_h: int):
    """Draw bounding boxes and IDs, rescaling coords from orig resolution to cell size."""
    h, w = frame.shape[:2]
    sx, sy = w / orig_w, h / orig_h
    for pid, bx, by, bw, bh in detections:
        col = color_for(pid)
        x1 = int(bx * sx)
        y1 = int(by * sy)
        x2 = int((bx + bw) * sx)
        y2 = int((by + bh) * sy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"ID:{pid}"
        # Place label above box, or inside if too close to top
        ly = y1 - 6 if y1 > 20 else y1 + 18
        draw_label(frame, label, x1, ly, col, font_scale=0.65, thickness=2)


def draw_legend(cell: np.ndarray, person_ids: list[int]):
    """Draw a colour legend in a cell."""
    cell[:] = (30, 30, 30)
    draw_label(cell, "Person ID Legend", 10, 30, (255, 255, 255), font_scale=0.8, thickness=2)
    for i, pid in enumerate(sorted(person_ids)):
        col = color_for(pid)
        y = 65 + i * 35
        cv2.rectangle(cell, (15, y - 12), (40, y + 8), col, -1)
        draw_label(cell, f"Person {pid}", 50, y + 5, col, font_scale=0.6, thickness=2)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mosaic tracking video.")
    parser.add_argument("--videos-dir", type=str, default=str(REPO_ROOT / "videos"),
                        help="Directory containing collated camera videos")
    parser.add_argument("--track-file", type=str, default=str(TRACK_FILE),
                        help="Path to track.txt")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help="Output mosaic video path")
    cli_args = parser.parse_args()

    videos_dir = Path(cli_args.videos_dir)
    track_file = Path(cli_args.track_file)
    output_file = Path(cli_args.output)

    # Discover cameras and their videos
    # IMPORTANT: prepare_data.py sorts video files lexicographically and assigns
    # sequential 1-based camera IDs (camera_0001, camera_0002, ...).  The tracker
    # uses these sequential IDs in track.txt.  We must reproduce the SAME ordering
    # so that track camera ID N maps to the correct video.
    video_files = sorted(
        p for p in videos_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    n_cams = len(video_files)

    # track_cam_id (1-based sequential) → video file path & display label
    cam_entries: list[dict] = []
    for idx, vp in enumerate(video_files, start=1):
        vid_label = vp.stem.split("_")[-1]  # e.g. "1", "10", …
        cam_entries.append({"track_cam": idx, "video": vp, "label": f"Cam {vid_label} (trk {idx})"})
    
    num_rows = (n_cams + 1 + NUM_COLS - 1) // NUM_COLS  # +1 for legend cell
    grid_w = NUM_COLS * CELL_W
    grid_h = num_rows * CELL_H

    print(f"Found {n_cams} cameras  →  {NUM_COLS}×{num_rows} grid  ({grid_w}×{grid_h})")
    for e in cam_entries:
        print(f"  track cam {e['track_cam']} → {e['video'].name}")

    # Load tracks
    tracks = load_tracks(track_file)
    all_pids = sorted({pid for dets in tracks.values() for pid, *_ in dets})
    print(f"Loaded tracks for {len(all_pids)} persons: {all_pids}")

    # Open video captures and get original resolutions
    caps: list[cv2.VideoCapture] = []
    orig_sizes: list[tuple[int, int]] = []

    for entry in cam_entries:
        cap = cv2.VideoCapture(str(entry["video"]))
        if not cap.isOpened():
            print(f"WARNING: cannot open {entry['video']}", file=sys.stderr)
        caps.append(cap)
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        orig_sizes.append((ow, oh))

    # Compute real FPS via ffprobe (immune to corrupted container metadata)
    global OUTPUT_FPS
    if OUTPUT_FPS is None:
        probed = [fps for e in cam_entries if (fps := _probe_fps(e["video"])) is not None]
        OUTPUT_FPS = float(np.median(probed)) if probed else 25.0
    print(f"Output FPS: {OUTPUT_FPS}")

    # Determine max frames across all videos
    max_frames = max(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps if c.isOpened())
    print(f"Max frames across cameras: {max_frames}")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_file), fourcc, OUTPUT_FPS, (grid_w, grid_h))
    if not writer.isOpened():
        print("ERROR: failed to create output video writer", file=sys.stderr)
        sys.exit(1)

    black_cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)

    # Pre-render legend cell
    legend_cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    draw_legend(legend_cell, all_pids)

    for fid in range(1, max_frames + 1):
        if fid % 100 == 1:
            print(f"  frame {fid}/{max_frames} …")

        mosaic = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx, (cap, (ow, oh)) in enumerate(zip(caps, orig_sizes)):
            ret, frame = cap.read()
            if ret and frame is not None:
                cell = cv2.resize(frame, (CELL_W, CELL_H))
            else:
                cell = black_cell.copy()

            # Draw camera label (show video name + track ID for clarity)
            entry = cam_entries[idx]
            draw_label(cell, entry["label"], 8, 28, (255, 255, 255), font_scale=0.65, thickness=2)

            # Draw tracking boxes using the sequential track camera ID
            key = (entry["track_cam"], fid)
            if key in tracks:
                draw_tracks_on_frame(cell, tracks[key], ow, oh)

            row, col = divmod(idx, NUM_COLS)
            y0, x0 = row * CELL_H, col * CELL_W
            mosaic[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell

        # Place legend in the cell right after the last camera
        legend_idx = n_cams
        lr, lc = divmod(legend_idx, NUM_COLS)
        if lr < num_rows:
            mosaic[lr * CELL_H:(lr + 1) * CELL_H, lc * CELL_W:(lc + 1) * CELL_W] = legend_cell

        writer.write(mosaic)

    writer.release()
    for c in caps:
        c.release()

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n✅  Mosaic saved → {output_file}  ({size_mb:.1f} MB, {max_frames} frames)")


if __name__ == "__main__":
    main()
