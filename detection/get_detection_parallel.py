import argparse
import multiprocessing as mp
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

def run_detection(args_tuple):
    cam, scene, ckpt, device, conf, batch, root_path = args_tuple
    cmd = [
        # Using sys.executable to ensure we use the same environment
        sys.executable,
        osp.join(root_path, "detection/get_detection_ultralytics.py"),
        "--scene", scene,
        "-c", ckpt,
        "--device", device,
        "--conf", str(conf),
        "--camera", cam,
        "--batchsize", str(batch),
    ]
    print(f"Starting {cam} on {device} with batch={batch}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene_001")
    parser.add_argument("-c", "--ckpt", type=str, default="yolo26m_people_detector.pt")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--procs-per-gpu", type=int, default=6)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    # Get absolute path to the root of the repo
    root_path = osp.abspath(osp.join(osp.dirname(__file__), ".."))
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
        # Round-robin assignment of cameras to GPUs
        gpu_idx = i % n_gpu
        device = devices[gpu_idx]
        work_queue.append((cam, args.scene, args.ckpt, device, args.conf, args.batch, root_path))

    print(f"Dispatching {len(cameras)} cameras to {total_procs} parallel processes on {args.devices}")
    
    with mp.Pool(min(len(cameras), total_procs)) as pool:
        pool.map(run_detection, work_queue)

if __name__ == "__main__":
    main()
