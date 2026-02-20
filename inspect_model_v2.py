from ultralytics import YOLO
import torch

model = YOLO('ckpt_weight/yolo12n_people_detector.pt')
print("--- Model Args ---")
for k, v in model.args.items():
    if 'mean' in k or 'std' in k or 'norm' in k or 'preprocess' in k:
        print(f"{k}: {v}")

print("\n--- Model Overrides ---")
print(model.overrides)

# Check for custom transforms in the underlying torch model
m = model.model
print(f"\nModel class: {m.__class__.__name__}")

# Try to see if there are any specific attributes in the model.pt
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

ckpt = torch.load('ckpt_weight/yolo12n_people_detector.pt', map_location='cpu', weights_only=False)
if 'model' in ckpt:
    m_inner = ckpt['model']
    print(f"\nInner model has args: {hasattr(m_inner, 'args')}")
    if hasattr(m_inner, 'args'):
        print(f"Inner args: {m_inner.args}")

if 'train_args' in ckpt:
    print(f"\nTrain args: {ckpt['train_args']}")
