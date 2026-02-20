#!/usr/bin/env python3
"""Run the full PoseTrack pipeline for every store under a source directory.

Each store directory is expected to contain one subdirectory per camera, with
chunk_*.mp4 files inside.  The script processes stores sequentially.

All artefacts are namespaced by store name so nothing is overwritten:
  - Collated videos  →  videos/<store_name>/
  - Dataset symlinks →  dataset/test/<store_name>/
  - Detection        →  result/detection/<store_name>/
  - ReID             →  result/reid/<store_name>/
  - Tracking         →  result/track/<store_name>.txt
  - Mosaic video     →  result/<store_name>/mosaic_tracks.mp4
  - Final track.txt  →  result/<store_name>/track.txt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/home/veesion/hq_cameras")


def banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}\n")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  → {' '.join(cmd[:6])}{'…' if len(cmd) > 6 else ''}")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}): {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def process_store(store_dir: Path) -> dict:
    """Run the full pipeline for a single store. Returns a summary dict."""
    store_name = store_dir.name
    banner(f"STORE: {store_name}")
    t0 = time.time()

    # ── Store-specific paths ──────────────────────────────────────────────
    videos_dir = REPO_ROOT / "videos" / store_name
    result_store_dir = REPO_ROOT / "result" / store_name
    track_scene_file = REPO_ROOT / "result" / "track" / f"{store_name}.txt"
    track_final = result_store_dir / "track.txt"
    mosaic_output = result_store_dir / "mosaic_tracks.mp4"

    result_store_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Collate camera chunks → videos/<store_name>/ ─────────────
    banner(f"[{store_name}] Step 0: Collate videos")
    run([
        sys.executable, "collate_videos.py",
        "--source", str(store_dir),
        "--dest", str(videos_dir),
    ])

    n_cameras = len(list(videos_dir.glob("*.mp4"))) if videos_dir.exists() else 0
    print(f"  Collated {n_cameras} camera videos")

    if n_cameras == 0:
        print(f"  WARNING: No videos collated for {store_name}, skipping.")
        return {"store": store_name, "cameras": 0, "status": "SKIPPED", "time": 0}

    # ── Steps 1-6: Full pipeline (scene = store_name) ────────────────────
    banner(f"[{store_name}] Steps 1-6: Full pipeline")
    run([
        sys.executable, "run_pipeline.py",
        "--scene", store_name,
        "--videos-dir", str(videos_dir),
    ])

    # ── Copy final track.txt to store-specific folder ────────────────────
    if track_scene_file.exists():
        shutil.copy2(track_scene_file, track_final)

    # ── Mosaic visualisation → result/<store_name>/mosaic_tracks.mp4 ────
    banner(f"[{store_name}] Step 7: Mosaic visualisation")
    run([
        sys.executable, "visualize_mosaic.py",
        "--videos-dir", str(videos_dir),
        "--track-file", str(track_final),
        "--output", str(mosaic_output),
    ])

    elapsed = time.time() - t0

    # ── Read summary from track file ─────────────────────────────────────
    n_people, n_dets = 0, 0
    if track_final.exists():
        import numpy as np
        data = np.loadtxt(str(track_final))
        if data.ndim == 2 and len(data) > 0:
            n_people = len(np.unique(data[:, 1]))
            n_dets = len(data)

    return {
        "store": store_name,
        "cameras": n_cameras,
        "people": n_people,
        "detections": n_dets,
        "time": elapsed,
        "status": "OK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PoseTrack pipeline for all stores."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help=f"Root directory containing store folders (default: {DEFAULT_SOURCE})",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"ERROR: Source directory does not exist: {source}", file=sys.stderr)
        sys.exit(1)

    stores = sorted([d for d in source.iterdir() if d.is_dir()])
    if not stores:
        print(f"ERROR: No store directories found in {source}", file=sys.stderr)
        sys.exit(1)

    banner(f"Processing {len(stores)} stores from {source}")
    for i, s in enumerate(stores, 1):
        n_cams = len([d for d in s.iterdir() if d.is_dir()])
        print(f"  {i}. {s.name}  ({n_cams} cameras)")

    total_t0 = time.time()
    summaries = []

    for store_dir in stores:
        summary = process_store(store_dir)
        summaries.append(summary)

    total_elapsed = time.time() - total_t0

    # ── Final summary ────────────────────────────────────────────────────
    banner("ALL STORES COMPLETE")
    print(f"{'Store':<40} {'Cams':>5} {'People':>7} {'Dets':>8} {'Time':>8} {'Status':>8}")
    print("-" * 80)
    for s in summaries:
        time_str = f"{s['time']/60:.1f}m" if s["time"] > 0 else "-"
        print(
            f"{s['store']:<40} {s.get('cameras',0):>5} "
            f"{s.get('people',0):>7} {s.get('detections',0):>8} "
            f"{time_str:>8} {s['status']:>8}"
        )
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("\nResults per store:")
    for s in summaries:
        print(f"  result/{s['store']}/track.txt")
        print(f"  result/{s['store']}/mosaic_tracks.mp4")


if __name__ == "__main__":
    main()
