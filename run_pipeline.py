from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SCENE_NAME = "scene_001"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    label = " ".join(cmd[:3]) + "..."
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def step_prepare() -> None:
    print("\n[1/6] Preparing dataset structure...")
    run([sys.executable, "prepare_data.py"])


def step_detection() -> None:
    print("\n[2/6] Running detection (YOLOX)...")
    run(
        [
            sys.executable,
            "detection/get_detection.py",
            "--scene",
            SCENE_NAME,
            "-f",
            "detection/yolox/exps/example/mot/yolox_x_mix_det.py",
            "-c",
            "ckpt_weight/bytetrack_x_mot17.pth.tar",
            "--device",
            "cpu",
            "--batchsize",
            "1",
        ]
    )


def step_pose() -> None:
    print("\n[3/6] Running pose estimation (HRNet)...")
    run(
        [
            sys.executable,
            "demo/save_pose_with_det_multiscene.py",
            "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
            "https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco",
            "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
            str(
                REPO_ROOT
                / "ckpt_weight"
                / "td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"
            ),
            "--input",
            "examples/88.jpg",
            "--output-root",
            "vis_results/",
            "--draw-bbox",
            "--show-kpt-idx",
            "--device",
            "cpu",
            "--start",
            "0",
            "--end",
            "1",
        ],
        cwd=REPO_ROOT / "mmpose",
    )


def step_reid() -> None:
    print("\n[4/6] Running ReID feature extraction...")
    run(
        [
            sys.executable,
            "tools/infer.py",
            "--start",
            "0",
            "--end",
            "1",
        ],
        cwd=REPO_ROOT / "fast-reid",
    )


def step_tracking() -> None:
    print("\n[5/6] Running multi-camera tracking...")
    result_dir = REPO_ROOT / "result"
    (result_dir / "track").mkdir(parents=True, exist_ok=True)
    (result_dir / "track_log").mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "track/run_tracking_batch.py",
            SCENE_NAME,
        ]
    )


def step_generate() -> None:
    print("\n[6/6] Generating final output...")
    run([sys.executable, "track/generate_submission.py"])
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
    steps = [
        step_prepare,
        step_detection,
        step_pose,
        step_reid,
        step_tracking,
        step_generate,
    ]
    for step_fn in steps:
        step_fn()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
