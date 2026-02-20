"""Drop-in replacement for get_detection.py using Ultralytics YOLO."""
from __future__ import annotations

import argparse
import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def make_parser():
    parser = argparse.ArgumentParser("Ultralytics YOLO Detection")
    parser.add_argument("--scene", type=str, default="scene_001",
                        help="scene directory name")
    parser.add_argument("-c", "--ckpt", type=str,
                        default="yolo26m_people_detector.pt",
                        help="path to YOLO checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="device: cpu, cuda:0, etc.")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="confidence threshold")
    parser.add_argument("--batchsize", type=int, default=1,
                        help="unused, kept for CLI compat")
    parser.add_argument("--camera", type=str, default=None,
                        help="specific camera to process (e.g. camera_0001)")
    return parser


def run_detection_inference(scene="scene_001", ckpt="yolo26m_people_detector.pt", device="cuda:0", conf=0.1, batchsize=1, camera=None):
    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split("/")[:-2]
    root_path = "/".join(path_arr)

    scene_name = scene
    input_dir = osp.join(root_path, "dataset/test", scene_name)
    out_path = osp.join(root_path, "result/detection", scene_name)
    os.makedirs(out_path, exist_ok=True)

    if device == "gpu":
        device = "cuda:0"

    model = YOLO(ckpt)
    print(f"Loaded model from {ckpt}")

    cameras = sorted(
        d for d in os.listdir(input_dir)
        if d.startswith("camera_")
    )
    if camera:
        if camera in cameras:
            cameras = [camera]
        else:
            print(f"ERROR: camera {camera} not found in {input_dir}")
            sys.exit(1)

    for cam in cameras:
        print(cam)
        video_path = osp.join(input_dir, cam, "video.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  WARNING: cannot open {video_path}, skipping")
            continue

        results_list: list[list] = []
        frame_id = 0
        pbar = tqdm(desc=cam)
        
        batch_size = batchsize
        batch_frames = []
        batch_ids = []

        while True:
            ret, frame = cap.read()
            if ret:
                batch_frames.append(frame)
                batch_ids.append(frame_id)
                frame_id += 1
            
            # Process batch if full or video ended
            if (not ret and batch_frames) or (len(batch_frames) >= batch_size):
                pbar.update(len(batch_frames))
                preds = model.predict(
                    batch_frames,
                    device=device,
                    conf=conf,
                    verbose=False,
                )

                for b_idx, r in enumerate(preds):
                    fid = batch_ids[b_idx]
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    xyxy = boxes.xyxy.cpu().numpy()   # (N,4) x1 y1 x2 y2
                    confs = boxes.conf.cpu().numpy()   # (N,)

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        score = float(confs[i])
                        # format: frame_id, cls, x1, y1, x2, y2, score
                        results_list.append(
                            [fid, 1,
                             int(x1), int(y1), int(x2), int(y2),
                             score]
                        )
                
                batch_frames = []
                batch_ids = []

            if not ret:
                break

        cap.release()
        pbar.close()

        output_file = osp.join(out_path, cam + ".txt")
        # Ensure results are sorted by frame_id
        results_list.sort(key=lambda x: x[0])
        with open(output_file, "w") as f:
            for row in results_list:
                f.write("{},{},{},{},{},{},{}\n".format(*row))
        print(f"  Wrote {len(results_list)} detections to {output_file}")


def main():
    args = make_parser().parse_args()
    run_detection_inference(args.scene, args.ckpt, args.device, args.conf, args.batchsize, args.camera)


if __name__ == "__main__":
    main()
