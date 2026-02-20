# PoseTrack Pipeline Setup & Test Run

- [x] Audit environment (Python, PyTorch, GPU, existing deps)
- [x] Install missing dependencies (mmcv, cvxopt, cython_bbox, faiss)
- [x] Download pretrained weights (4 files via gdown)
- [x] Collate camera videos: `python3 collate_videos.py [--source DIR]`. This script merges chunks from `/home/veesion/hq_cameras/au-iga-4169-lytton-33/` (default) while preserving timestamps for sync. Resulting videos are in `videos/`.
- [x] Upgrade ReID model: replaced fast-reid MGN+R101-IBN → CLIP-ReID → **SOLIDER-REID (Swin-Base, MSMT17)** — CVPR 2023 semantic-controllable SSL
- [x] Fix tracking: rebuild aic_cpp, np.float deprecation, NaN guards
- [x] Add `--start-step` to `run_pipeline.py` to skip completed steps
- [x] Run full pipeline ✅
- [x] Create mosaic visualization for results 🎞️


## Results

**4 people** tracked, **4681 detections** across **7 cameras**, frames 2–4497. *(SOLIDER-REID Swin-Base, MSMT17)*

Output: `result/track.txt`

## Pipeline steps

Use `python3` of the machine.

| Step | Name | Command to resume from |
|------|------|-----------------------|
| 1 | Prepare dataset | `python3 run_pipeline.py --start-step 1` |
| 2 | Detection (YOLO) | `python3 run_pipeline.py --start-step 2` |
| 3 | ReID features (SOLIDER-REID Swin-Base) | `python3 run_pipeline.py --start-step 3` |
| 4 | Multi-camera tracking | `python3 run_pipeline.py --start-step 4` |
| 5 | Generate output | `python3 run_pipeline.py --start-step 5` |
| 6 | Visualize results | `python3 visualize_mosaic.py` |

## Multi-store batch run

Runs the full pipeline for every store under a source path. Each store's artifacts are fully namespaced so nothing is shared or overwritten.

```bash
python3 run_all_stores.py [--source /home/veesion/hq_cameras]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--source` | `/home/veesion/hq_cameras` | Root directory containing store folders |

**Directory layout per store** (using store name as scene name):

| Artifact | Path |
|----------|------|
| Collated videos | `videos/<store>/` |
| Dataset symlinks | `dataset/test/<store>/` |
| Detections | `result/detection/<store>/` |
| ReID features | `result/reid/<store>/` |
| Track (scene) | `result/track/<store>.txt` |
| Track (final) | `result/<store>/track.txt` |
| Mosaic video | `result/<store>/mosaic_tracks.mp4` |

All pipeline scripts (`run_pipeline.py`, `prepare_data.py`, `collate_videos.py`, `visualize_mosaic.py`) now accept `--scene` / `--videos-dir` / `--dest` / `--track-file` / `--output` arguments for manual per-store runs.

## Standalone: ByteTrack per-camera visualization

Runs YOLO detection + ByteTrack tracking from scratch on raw camera chunks and exports annotated videos (bbox + track ID overlay). Does **not** use any previously exported tracks.

```bash
python3 bytetrack_visualize.py --source /home/veesion/hq_cameras/<store-name>
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--source` | *(required)* | Store directory containing camera folders with `chunk_*.mp4` files |
| `--model` | `yolo26m_people_detector.pt` | YOLO model checkpoint |
| `--device` | `cuda:0` | GPU device |
| `--conf` | `0.3` | Detection confidence threshold |
| `--output` | `result/<store>/bytetrack_videos/` | Output directory for annotated videos |

- Collates chunks into `collated_tmp/<store>/` (not `videos/`), reuses if already collated
- Skips cameras whose output video already exists
- One output mp4 per camera with colored bounding boxes and track IDs
