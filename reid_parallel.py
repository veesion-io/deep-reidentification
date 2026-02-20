"""Parallel SOLIDER-REID feature extraction — dispatches cameras across GPUs."""
import argparse
import multiprocessing as mp
import os
import os.path as osp
import subprocess
import sys


def run_reid(args_tuple):
    from solider_reid_infer import run_reid_inference
    import torch
    
    cam, scene, device, root_path = args_tuple
    save_path = osp.join(root_path, "result/reid", scene, cam + ".npy")
    if osp.exists(save_path):
        print(f"  Skipping {cam} (already exists)")
        return

    # Use CUDA_VISIBLE_DEVICES instead of passing string to torch to avoid multiprocessing context issues if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

    print(f"Starting ReID for {cam} on {device}")
    
    # We pass device internally, Solider-ReID inference will pick up the visible device
    run_reid_inference(scene=scene, camera=cam)


def run_parallel_reid(scene="scene_001", procs_per_gpu=5, devices="cuda:0,cuda:1"):
    root_path = osp.abspath(osp.join(osp.dirname(__file__)))
    input_dir = osp.join(root_path, "dataset/test", scene)

    cameras = sorted([
        d for d in os.listdir(input_dir)
        if d.startswith("camera_")
    ])

    device_list = devices.split(",")
    n_gpu = len(device_list)
    total_procs = n_gpu * procs_per_gpu

    work_queue = []
    for i, cam in enumerate(cameras):
        gpu_idx = i % n_gpu
        device = device_list[gpu_idx]
        work_queue.append((cam, scene, device, root_path))

    print(f"Dispatching {len(cameras)} cameras to {total_procs} parallel processes on {devices}")

    ctx = mp.get_context('spawn')
    with ctx.Pool(min(len(cameras), total_procs)) as pool:
        pool.map(run_reid, work_queue)

    print("All ReID feature extraction done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene_001")
    parser.add_argument("--procs-per-gpu", type=int, default=5)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    args = parser.parse_args()

    run_parallel_reid(args.scene, args.procs_per_gpu, args.devices)


if __name__ == "__main__":
    main()
