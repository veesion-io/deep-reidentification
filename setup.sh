#!/usr/bin/env bash
set -euo pipefail

echo "=== PoseTrack Environment Setup ==="

# 1. Create conda env (CPU-only PyTorch)
if ! conda env list | grep -q "aic24"; then
    echo "Creating conda environment 'aic24'..."
    conda create -n aic24 python=3.9 pytorch torchvision cpuonly -c pytorch -y
else
    echo "Conda env 'aic24' already exists, skipping creation."
fi

echo "Activate with: conda activate aic24"
echo "Then run the remaining setup steps below."

# The rest should be run inside the conda env:
# pip install -r requirements.txt
# pip install -e track/aic_cpp
# pip install openmim
# mim install "mmengine>=0.6.0"
# mim install "mmcv==2.0.1"
# mim install "mmpose>=1.1.0"
# pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# 2. Download pretrained weights
# pip install gdown
# mkdir -p ckpt_weight
# gdown 1LVFqYqx88R0TUjCMbTaKrJkL7-SdCSmC -O ckpt_weight/bytetrack_x_mot17.pth.tar
# gdown 1xDKWJRWja01nNOeV7TWcn58sHYSal2k9 -O ckpt_weight/luperson_resnet.pth
# gdown 1tNT6gOBB95qYPCypvCctj1o-r7bzdwxA -O ckpt_weight/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth
# gdown 17qbBmBX7DiT2lOuQ6rGHl8s9deKHkVn2 -O ckpt_weight/aic24.pkl
