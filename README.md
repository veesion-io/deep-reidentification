# Veesion Multi camera tracking Pipeline

This repository contains the Veesion multi-camera tracking pipeline. The primary purpose of this codebase is to process store camera videos to perform multi-camera people tracking, object detection, ReID feature extraction, and track visualization.

## Online Service Pipeline

The repository has been adapted to be usable in an online fashion, executing live in the cloud on a continuous instance. It is designed to receive a request containing an S3 videos directory and a JSON payload, and then perform the following steps:

1. **Download Data**: Download all video data from the specified S3 directory. The directory contains multiple folders (one for each camera from a store), and each folder contains multiple chunks of videos.
   
   **Expected S3 Directory Layout:**
   ```
   s3://your-bucket/.../<store-directory>/
   ├── camera_0001/
   │   ├── chunk_0000.mp4
   │   ├── chunk_0001.mp4
   │   └── ...
   ├── camera_0002/
   │   ├── chunk_0000.mp4
   │   └── ...
   ```
2. **Collate Videos**: Collate the video chunks into full videos, carefully preserving their original timestamp information.
3. **Run Core Models**: Execute the object detection, multi-camera tracking, and ReID models on the collated videos.
4. **Identify Query Track**: Use the provided JSON, which contains the info of the track of the person to be tracked across multiple cameras. The specific track is identified based on spatio-temporal IOU with our generated tracks.

   **Expected JSON Format for Query Track:**
   ```json
   {
     "camera": "camera_0001",
     "track": [
       {
         "frame_id": 10,
         "bbox": [100, 200, 50, 150]
       },
       {
         "frame_id": 11,
         "bbox": [110, 205, 50, 150]
       }
     ]
   }
   ```
   *(Note: The `bbox` format is `[x, y, w, h]`)*

5. **Semi-Automatic Matching**: Run multi-camera matching in a semi-automatic fashion. This returns a list of potential pairs associated with the query track, ordered by decreasing likelihood (for all pairs with a ReID score above 0.5).
6. **Generate Video Pairs**: For each potential pair, generate a pair video (using minimal computation) comparing the query track (left) and the candidate track (right), with bounding box visualizations.
7. **Human Annotation (Tier Service)**: Send the generated video pairs to a tier service (TBD) and synchronously await human annotation confirming or rejecting the pairs.
8. **Construct Final Timeline**: Once all pairs are annotated, build the complete, continuous video timeline of the query track from start to finish. If the person appears in multiple videos simultaneously, the video feeds are collated side-by-side without deformation (e.g., jumping between cameras or showing split screens when overlapping).
9. **Client Alerting**: Send the final timeline video to another service for client alerting (TBD).

## Environment & Setup

Ensure you are running on a machine with Python 3, PyTorch, and a CUDA-compatible GPU.

The required dependencies include:
- `mmcv`
- `cvxopt`
- `cython_bbox`
- `faiss`
- Standard data science / CV Python packages

You also must have the pretrained weights downloaded (4 files via gdown). The models rely on:
- YOLOv26m for detection
- **SOLIDER-REID (Swin-Base, MSMT17)** for multi-camera ReID matching (CVPR 2023 semantic-controllable SSL)

## Pipeline Steps

To run the pipeline on a single store, use the machine's `python3`. The pipeline consists of multiple steps that can be resumed natively using `--start-step`.

| Step | Name | Command to Resume/Run |
|------|------|-----------------------|
| 1 | Prepare dataset | `python3 pipeline/run_pipeline.py --start-step 1` |
| 2 | Detection (YOLO) | `python3 pipeline/run_pipeline.py --start-step 2` |
| 3 | ReID features (SOLIDER-REID) | `python3 pipeline/run_pipeline.py --start-step 3` |
| 4 | Multi-camera tracking | `python3 pipeline/run_pipeline.py --start-step 4` |
| 5 | Generate output | `python3 pipeline/run_pipeline.py --start-step 5` |
| 6 | Visualize results | `python3 pipeline/visualize_mosaic.py` |

**Pre-Step: Collating Videos**
Before running the main pipeline, if your videos are in chunks, you must collate them while preserving timestamps for synchronization:
```bash
python3 pipeline/collate_videos.py [--source /path/to/store/cameras]
```
*(Default source is `/home/veesion/hq_cameras/au-iga-4169-lytton-33/`. Resulting videos are placed in `videos/`.)*

## Multi-Store Batch Run

You can run the full pipeline automatically for every store directory located under a source path. Each store's artifacts are fully namespaced to prevent overwriting.

```bash
python3 pipeline/run_all_stores.py [--source /home/veesion/hq_cameras]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `/home/veesion/hq_cameras` | Root directory containing store folders |

### Directory Layout per Store
When running batch processing (or single-store with explicit args), the pipeline uses the folder name as the `<store>` pseudo-scene name:

| Artifact | Path |
|----------|------|
| Collated videos | `videos/<store>/` |
| Dataset symlinks | `dataset/test/<store>/` |
| Detections | `result/detection/<store>/` |
| ReID features | `result/reid/<store>/` |
| Track (scene) | `result/track/<store>.txt` |
| Track (final) | `result/<store>/track.txt` |
| Mosaic video | `result/<store>/mosaic_tracks.mp4` |

*Note: All core pipeline scripts (`pipeline/run_pipeline.py`, `pipeline/prepare_data.py`, `pipeline/collate_videos.py`, `pipeline/visualize_mosaic.py`) natively accept `--scene`, `--videos-dir`, `--dest`, `--track-file`, `--output` parameters for advanced manual invocation per-store.*

## Standalone ByteTrack Per-Camera Visualization

If you want to run YOLO detection and ByteTrack tracking *from scratch* on raw camera chunks and export annotated videos (bounding boxes + track IDs), use the standalone visualizer. This does **not** use the globally exported multi-camera tracks.

```bash
python3 pipeline/bytetrack_visualize.py --source /home/veesion/hq_cameras/<store-name>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | *(required)* | Store dir containing camera folders with `chunk_*.mp4` |
| `--model` | `yolo26m_people_detector.pt`| YOLO model checkpoint to use |
| `--device` | `cuda:0` | GPU device |
| `--conf` | `0.3` | Detection confidence threshold |
| `--output` | `result/<store>/bytetrack_videos/` | Output directory for annotated videos |

- Reuses existing collated files in `collated_tmp/<store>/` if they exist.
- Safely skips processing cameras whose output `.mp4` already exists.
- Generates one `.mp4` per camera.

## Latest Tracking Test Results

For reference, running this pipeline on the `au-iga-4169-lytton-33` configuration setup (frames 2–4497):
- **4 people** accurately tracked.
- **4681 detections** matched across **7 distinct cameras**.
- Overall output generated reliably at: `result/track.txt`