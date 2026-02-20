from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO('ckpt_weight/yolo12n_people_detector.pt')
print(f"Model internal imgsz: {model.args['imgsz']}")

# Load a sample frame from the dataset if available
video_path = "dataset/test/scene_001/camera_0001/video.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"Input frame shape: {frame.shape}")
        # Run prediction
        results = model.predict(frame, verbose=True)
        
        # Check if results are scaled back to original resolution
        for r in results:
            print(f"Output box format: {r.boxes.xyxy[0] if len(r.boxes) > 0 else 'No detections'}")
            # The 'orig_shape' attribute shows the original frame size
            print(f"Result orig_shape: {r.orig_shape}")
            
            # We can also check the preprocessing shape if available in newer versions
            # but usually imgsz in verbose output is enough.
    else:
        print("Failed to read frame from video.")
else:
    print(f"Video not found at {video_path}")
