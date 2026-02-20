"""Parallel SOLIDER-REID feature extraction — dispatches cameras across GPUs."""
import argparse
import multiprocessing as mp
import os
import os.path as osp
import subprocess
import sys


def run_reid(args_tuple):
    cam, scene, device, root_path = args_tuple
    save_path = osp.join(root_path, "result/reid", scene, cam + ".npy")
    if osp.exists(save_path):
        print(f"  Skipping {cam} (already exists)")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

    cmd = [
        sys.executable,
        osp.join(root_path, "solider_reid_infer.py"),
        "--scene", scene,
        "--camera", cam,
    ]
    print(f"Starting ReID for {cam} on {device}")
    subprocess.run(cmd, check=True, cwd=root_path, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene_001")
    parser.add_argument("--procs-per-gpu", type=int, default=5)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    args = parser.parse_args()

    root_path = osp.abspath(osp.join(osp.dirname(__file__)))
    input_dir = osp.join(root_path, "dataset/test", args.scene)

    cameras = sorted([
        d for d in os.listdir(input_dir)
        if d.startswith("camera_")
    ])

    devices = args.devices.split(",")
    n_gpu = len(devices)
    total_procs = n_gpu * args.procs_per_gpu

    work_queue = []
    for i, cam in enumerate(cameras):
        gpu_idx = i % n_gpu
        device = devices[gpu_idx]
        work_queue.append((cam, args.scene, device, root_path))

    print(f"Dispatching {len(cameras)} cameras to {total_procs} parallel processes on {args.devices}")

    with mp.Pool(min(len(cameras), total_procs)) as pool:
        pool.map(run_reid, work_queue)

    print("All ReID feature extraction done.")


if __name__ == "__main__":
    main()
