from ultralytics import YOLO
import torch

model = YOLO('ckpt_weight/yolo12n_people_detector.pt')
print(f"Model Task: {model.task}")
print(f"Model Names: {model.names}")
print(f"Model Overrides: {model.overrides}")

# Check default predict arguments
print(f"Default Predict Args: {model.args}")

# Check imgsz specifically
print(f"ImgSz in model.args: {model.args.get('imgsz')}")
