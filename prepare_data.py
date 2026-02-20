from __future__ import annotations

import argparse

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = REPO_ROOT / "videos"
DATASET_DIR = REPO_ROOT / "dataset" / "test" / "scene_001"


def make_identity_calibration() -> dict[str, list[list[float]]]:
    projection = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    homography = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return {
        "camera projection matrix": projection,
        "homography matrix": homography,
    }


def prepare(videos_dir: Path = VIDEOS_DIR, dataset_dir: Path = DATASET_DIR) -> None:
    if dataset_dir.exists():
        import shutil
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p
        for p in videos_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if not video_files:
        print(f"No video files found in {videos_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(video_files)} videos:")
    calibration = make_identity_calibration()

    for idx, video_path in enumerate(video_files, start=1):
        cam_name = f"camera_{idx:04d}"
        cam_dir = dataset_dir / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = cam_dir / "video.mp4"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(video_path.resolve())

        cal_path = cam_dir / "calibration.json"
        cal_path.write_text(json.dumps(calibration, indent=2))

        print(f"  {cam_name} -> {video_path.name}")

    print(f"\nDataset prepared at {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=str, default=str(VIDEOS_DIR))
    parser.add_argument("--scene", type=str, default="scene_001")
    args = parser.parse_args()
    vdir = Path(args.videos_dir)
    ddir = REPO_ROOT / "dataset" / "test" / args.scene
    prepare(videos_dir=vdir, dataset_dir=ddir)
