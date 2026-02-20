#!/usr/bin/env python3
"""
SOLIDER-REID feature extraction for PoseTrack pipeline.
Drop-in replacement for clip_reid_infer.py using the SOLIDER-REID
(Swin-Base) model — CVPR 2023, semantic-controllable self-supervised learning.

Reads detections from result/detection/, video frames from dataset/test/,
and saves per-camera .npy feature files to result/reid/.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import mmcv
from argparse import ArgumentParser
from tqdm import tqdm

# ── Locate project root ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = SCRIPT_DIR  # solider_reid_infer.py lives at repo root

# Add SOLIDER-REID to path
sys.path.insert(0, os.path.join(ROOT_PATH, "SOLIDER-REID"))


# ── Feature dimension for Swin-Base ──
FEAT_DIM = 1024  # Swin-Base: embed_dims=128, 4 stages → 128*8=1024


def build_solider_reid_model(weight_path, device="cuda"):
    """Build and load a SOLIDER-REID Swin-Base model from a fine-tuned checkpoint."""
    from config import cfg
    from model import make_model

    # Set config values programmatically for Swin-Base inference
    cfg.merge_from_list([
        "MODEL.NAME", "transformer",
        "MODEL.TRANSFORMER_TYPE", "swin_base_patch4_window7_224",
        "MODEL.PRETRAIN_CHOICE", "self",
        "MODEL.PRETRAIN_PATH", "",
        "MODEL.PRETRAIN_HW_RATIO", "2",
        "MODEL.STRIDE_SIZE", "[16, 16]",
        "MODEL.SIE_CAMERA", "False",
        "MODEL.SIE_VIEW", "False",
        "MODEL.SIE_COE", "3.0",
        "MODEL.COS_LAYER", "False",
        "MODEL.NECK", "bnneck",
        "MODEL.REDUCE_FEAT_DIM", "False",
        "MODEL.FEAT_DIM", "512",
        "MODEL.DROPOUT_RATE", "0.0",
        "MODEL.DROP_PATH", "0.1",
        "MODEL.DROP_OUT", "0.0",
        "MODEL.ATT_DROP_RATE", "0.0",
        "MODEL.JPM", "False",
        "MODEL.SEMANTIC_WEIGHT", "0.2",
        "INPUT.SIZE_TRAIN", "[384, 128]",
        "INPUT.SIZE_TEST", "[384, 128]",
        "INPUT.PIXEL_MEAN", "[0.5, 0.5, 0.5]",
        "INPUT.PIXEL_STD", "[0.5, 0.5, 0.5]",
        "TEST.NECK_FEAT", "before",
        "TEST.FEAT_NORM", "yes",
    ])
    cfg.freeze()

    # MSMT17 has 1041 training identities (used for classifier layer, not for inference)
    num_classes = 1041
    camera_num = 0
    view_num = 0
    semantic_weight = 0.2

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=semantic_weight)
    model.load_param(weight_path)
    model = model.to(device)
    model.eval()
    return model, cfg


class SOLIDERReIDInferencer:
    """Extract SOLIDER-REID features from person crops."""

    def __init__(self, model, cfg, device="cuda"):
        self.model = model
        self.device = device
        self.mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1).to(device)
        self.std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1).to(device)
        self.input_h = cfg.INPUT.SIZE_TEST[0]  # 384
        self.input_w = cfg.INPUT.SIZE_TEST[1]  # 128

    @torch.no_grad()
    def process_frame(self, frame, bboxes):
        """
        Extract ReID features for all bounding boxes in a frame.

        Args:
            frame: BGR numpy array (H, W, 3)
            bboxes: (N, 4) array with columns [x1, y1, x2, y2]

        Returns:
            features: (N, FEAT_DIM) numpy array of L2-normalized features
        """
        if len(bboxes) == 0:
            return np.zeros((0, FEAT_DIM), dtype=np.float32)

        # Convert BGR to RGB and to float tensor
        frame_rgb = frame[:, :, ::-1].copy()
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                # Degenerate box → create a tiny dummy crop
                crop = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
            else:
                crop = frame_rgb[y1:y2, x1:x2]
                crop = cv2.resize(crop, (self.input_w, self.input_h))
            crops.append(crop)

        # Stack and normalize: (N, 3, H, W)
        crops = np.stack(crops, axis=0).astype(np.float32) / 255.0
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).to(self.device)
        crops = (crops - self.mean) / self.std

        # Forward pass → (feat, featmaps); take feat which is (N, FEAT_DIM)
        output = self.model(crops)
        if isinstance(output, tuple):
            feats = output[0]  # global features
        else:
            feats = output

        # Also do horizontal flip augmentation (test-time augmentation)
        output_flip = self.model(crops.flip(3))
        if isinstance(output_flip, tuple):
            feats_flip = output_flip[0]
        else:
            feats_flip = output_flip

        feats = (feats + feats_flip) / 2.0

        # L2 normalize
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy()


def run_reid_inference(scene=None, camera=None, start=0, end=-1, batch_size=64):
    det_root = os.path.join(ROOT_PATH, "result/detection")
    vid_root = os.path.join(ROOT_PATH, "dataset/test")
    save_root = os.path.join(ROOT_PATH, "result/reid")

    if scene:
        scenes = [scene]
    else:
        scenes = sorted(os.listdir(det_root))
        scenes = [s for s in scenes if not s.startswith(".")]
        scenes = scenes[start : end if end != -1 else None]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_path = os.path.join(ROOT_PATH, "transformer_100.pth")
    print(f"Loading SOLIDER-REID (Swin-Base, MSMT17) from {weight_path}...")
    model, cfg = build_solider_reid_model(weight_path, device)
    inferencer = SOLIDERReIDInferencer(model, cfg, device)
    print(f"Model loaded successfully. Feature dim: {FEAT_DIM}")

    for s in tqdm(scenes, desc="Scenes"):
        print(s)
        det_dir = os.path.join(det_root, s)
        vid_dir = os.path.join(vid_root, s)
        save_dir = os.path.join(save_root, s)
        cams = os.listdir(vid_dir)
        cams = sorted([c for c in cams if c[0] == "c" or c.startswith("camera_")])

        os.makedirs(save_dir, exist_ok=True)

        # If camera is specified, process only that camera
        if camera:
            cams = [camera]

        print(f"  {len(cams)} cameras")
        for cam in tqdm(cams, desc="Cameras"):
            print(f"  {cam}")
            det_path = os.path.join(det_dir, cam) + ".txt"
            vid_path = os.path.join(vid_dir, cam, "video.mp4")
            save_path = os.path.join(save_dir, cam + ".npy")

            if os.path.exists(save_path):
                print(f"    Skipping (already exists)")
                continue

            # Load detections
            if not os.path.exists(det_path):
                print(f"    Skipping {cam}: no detection file")
                np.save(save_path, np.array([]))
                continue

            det_annot = np.ascontiguousarray(np.loadtxt(det_path, delimiter=","))
            if det_annot.ndim < 2 or len(det_annot) == 0:
                print(f"    Skipping {cam}: no detections")
                np.save(save_path, np.array([]))
                continue

            # Read video
            video = mmcv.VideoReader(vid_path)
            screen_height, screen_width = video.height, video.width

            all_results = []
            for frame_id, frame in enumerate(tqdm(video, total=len(video), desc=cam)):
                dets = det_annot[det_annot[:, 0] == frame_id]
                bboxes_s = dets[:, 2:7]  # x1, y1, x2, y2, score

                if len(bboxes_s) == 0:
                    continue

                # Clamp to frame bounds
                bboxes = bboxes_s[:, :4].copy()
                bboxes[:, 0] = np.maximum(0, bboxes[:, 0])
                bboxes[:, 1] = np.maximum(0, bboxes[:, 1])
                bboxes[:, 2] = np.minimum(screen_width, bboxes[:, 2])
                bboxes[:, 3] = np.minimum(screen_height, bboxes[:, 3])

                # Process in batches
                n = len(bboxes)
                batch_feats = []
                for i in range(0, n, batch_size):
                    batch = bboxes[i : i + batch_size]
                    feats = inferencer.process_frame(frame, batch)
                    batch_feats.append(feats)
                feat_combined = np.concatenate(batch_feats, axis=0)
                all_results.append(feat_combined)

            if len(all_results) == 0:
                all_results = np.array([])
            else:
                all_results = np.concatenate(all_results)

            np.save(save_path, all_results)
            print(f"    Saved {save_path}: {all_results.shape}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for feature extraction")
    parser.add_argument("--camera", type=str, default=None,
                        help="Process a single camera (used by reid_parallel.py)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process a single scene (used by reid_parallel.py)")
    args = parser.parse_args()
    
    run_reid_inference(args.scene, args.camera, args.start, args.end, args.batch_size)




if __name__ == "__main__":
    main()
