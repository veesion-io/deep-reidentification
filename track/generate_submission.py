import argparse
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate final track.txt from scene track files.")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name to generate submission for (default: all scenes)")
    args = parser.parse_args()

    result_dir = "result/track"
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)

    if args.scene:
        # Single scene mode — look for <scene>.txt
        scene_file = osp.join(result_dir, args.scene + ".txt")
        if not osp.exists(scene_file):
            print(f"ERROR: Track file not found: {scene_file}")
            return
        scenes = [args.scene + ".txt"]
    else:
        # All scenes — find all .txt files (excluding _tracklets files)
        scenes = sorted([s for s in os.listdir(result_dir)
                         if s.endswith(".txt") and "_tracklets" not in s])

    if not scenes:
        print(f"ERROR: No track files found in {result_dir}")
        return

    files = []
    for s in tqdm(scenes):
        data = np.loadtxt(osp.join(result_dir, s))
        if data.ndim == 2 and len(data) > 0:
            files.append(data[:, :-1])

    if not files:
        print("ERROR: No valid track data found.")
        return

    results = np.concatenate(files, axis=0)
    np.savetxt(osp.join(save_dir, "track.txt"), results, fmt="%d %d %d %d %d %d %d %f %f")
    print(f"Generated {save_dir}/track.txt with {len(results)} detections from {len(files)} scene(s)")


if __name__ == "__main__":
    main()

