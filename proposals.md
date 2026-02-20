## Human-in-the-Loop Match Proposals

Generate clean single-camera tracklets (no ReID-based re-identification), then compute cross-camera match proposals ranked by cosine similarity for human review.

### 1. Generate clean tracklets

```bash
cd track && python3 run_tracking_batch.py scene_001 --no-reid-merge
```

This disables `match_with_miss_tracks` in the tracker, producing clean continuous tracklets saved to `result/track/scene_001_tracklets.txt`. Each tracklet is a continuous detection segment on a single camera — no ReID gap-bridging.

### 2. Generate match proposals

```bash
python3 match_proposals.py --top 20
```

Computes pairwise ReID cosine similarity between all cross-camera tracklet pairs, ranks them, and generates side-by-side cropped comparison videos in `result/match_proposals/`. Each video shows both tracklets temporally aligned with bounding box overlays.

Options:
- `--top N` — number of top proposals to visualize (default: 15)
- `--crop-size PX` — crop square size in pixels (default: 250)
- `--track-file PATH` — path to track/tracklet file (default: `result/track/scene_001_tracklets.txt`)

See [graph.md](graph.md) for the design of an interactive annotation system with active learning.