from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_NAME = "scene_001"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    label = " ".join(cmd[:3]) + "..."
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def step_prepare(videos_dir: str | None = None, scene: str = DEFAULT_SCENE_NAME) -> None:
    print("\n[1/6] Preparing dataset structure...")
    cmd = [sys.executable, "pipeline/prepare_data.py", "--scene", scene]
    if videos_dir:
        cmd += ["--videos-dir", videos_dir]
    run(cmd)

def step_detection(scene: str = DEFAULT_SCENE_NAME, conf: float = 0.1, batch: int = 16) -> None:
    print(f"\n[2/6] Running parallel detection with conf={conf}, batch={batch}...")
    run(
        [
            sys.executable,
            "detection/get_detection_parallel.py",
            "--scene",
            scene,
            "--conf",
            str(conf),
            "--batch",
            str(batch),
        ]
    )

def step_reid(scene: str = DEFAULT_SCENE_NAME) -> None:
    print("\n[3/5] Running ReID feature extraction (SOLIDER-REID Swin-Base, parallel)...")
    run(
        [
            sys.executable,
            "reid_parallel.py",
            "--scene", scene,
            "--procs-per-gpu", "5",
            "--devices", "cuda:0,cuda:1",
        ],
    )


def step_tracking(scene: str = DEFAULT_SCENE_NAME) -> None:
    print("\n[4/5] Running multi-camera tracking...")
    result_dir = REPO_ROOT / "result"
    (result_dir / "track").mkdir(parents=True, exist_ok=True)
    (result_dir / "track_log").mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "track/run_tracking_batch.py",
            scene,
        ]
    )


def step_generate(scene: str = DEFAULT_SCENE_NAME) -> None:
    print("\n[5/5] Generating final output...")
    run([sys.executable, "track/generate_submission.py", "--scene", scene])
    track_file = REPO_ROOT / "result" / "track.txt"
    if track_file.exists():
        import numpy as np

        data = np.loadtxt(str(track_file))
        n_tracks = len(np.unique(data[:, 1]))
        n_dets = len(data)
        cams = np.unique(data[:, 0])
        print(
            f"\nResults: {n_tracks} people, {n_dets} detections across {len(cams)} cameras"
        )
        print(f"Cameras: {cams}")
        print(f"Frame range: {data[:, 2].min():.0f} - {data[:, 2].max():.0f}")
    else:
        print("WARNING: result/track.txt not found!", file=sys.stderr)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="PoseTrack pipeline")
    parser.add_argument(
        "--start-step", type=int, default=1,
        help="Step to start from (1-5): 1=prepare, 2=detection, 3=reid, 4=tracking, 5=generate",
    )
    parser.add_argument(
        "--scene", type=str, default=DEFAULT_SCENE_NAME,
        help="Scene name — used to namespace all intermediate results (default: scene_001)",
    )
    parser.add_argument(
        "--videos-dir", type=str, default=None,
        help="Directory containing collated videos (default: videos/)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.1,
        help="Confidence threshold for detection (default: 0.1)",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size for detection (default: 16)",
    )
    args = parser.parse_args()

    steps = [
        (1, lambda: step_prepare(args.videos_dir, args.scene)),
        (2, lambda: step_detection(args.scene, args.conf, args.batch)),
        (3, lambda: step_reid(args.scene)),
        (4, lambda: step_tracking(args.scene)),
        (5, lambda: step_generate(args.scene)),
    ]
    for i, step_fn in steps:
        if i < args.start_step:
            continue
        step_fn()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
