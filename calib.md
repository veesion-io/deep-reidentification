## CasCalib Integration

To run camera self-calibration using CasCalib:

1. **Convert Detections**: Run `python3 convert_to_cascalib.py`. This script extracts tracks from `result/track.txt` and poses from `result/pose/scene_001/*.txt`, converts them to the CasCalib JSON format, and prepares dummy frames for resolution metadata.
2. **Run Calibration**: 
   ```bash
   cd CasCalib
   python3 run_cascalib_posetrack.py
   ```
   This will:
   - Perform single-view calibration for each camera (focal length and ground plane estimation).
   - Perform multi-view temporal and spatial alignment.
   - Save the results to `calib_results_cascalib.json` in the root directory.

> [!NOTE]
> The current calibration is not yet integrated into the ReID/matching step of the main pipeline. It operates as an independent post-processing or analysis step.

