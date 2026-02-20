import argparse
import multiprocessing as mp
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

def run_detection(args_tuple):
    from detection.get_detection_ultralytics import run_detection_inference
    cam, scene, ckpt, device, conf, batch, root_path = args_tuple
    print(f"Starting {cam} on {device} with batch={batch}")
    run_detection_inference(scene=scene, ckpt=ckpt, device=device, conf=conf, batchsize=batch, camera=cam)

def run_parallel_detection(scene="scene_001", ckpt="yolo26m_people_detector.pt", conf=0.7, procs_per_gpu=6, devices="cuda:0,cuda:1", batch=16):
    # Get absolute path to the root of the repo
    root_path = osp.abspath(osp.join(osp.dirname(__file__), ".."))
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
        # Round-robin assignment of cameras to GPUs
        gpu_idx = i % n_gpu
        device = device_list[gpu_idx]
        work_queue.append((cam, scene, ckpt, device, conf, batch, root_path))

    print(f"Dispatching {len(cameras)} cameras to {total_procs} parallel processes on {devices}")
    
    # Needs context='spawn' for YOLO CUDA compatibility in multiprocessing
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(len(cameras), total_procs)) as pool:
        pool.map(run_detection, work_queue)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene_001")
    parser.add_argument("-c", "--ckpt", type=str, default="yolo26m_people_detector.pt")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--procs-per-gpu", type=int, default=6)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    run_parallel_detection(args.scene, args.ckpt, args.conf, args.procs_per_gpu, args.devices, args.batch)

if __name__ == "__main__":
    main()
