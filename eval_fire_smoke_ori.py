from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face
model_path = hf_hub_download(
    repo_id="TommyNgx/YOLOv10-Fire-and-Smoke-Detection",
    filename="best.pt",
)

# Initialize the YOLO model
model = YOLO(model_path)

# Run inference
results = model.predict("/home/zhaorun/zichen/yjb/projects/CV/MambaIC/dataset/wildfire/all/16M00020220317135_20230626170015.jpg", conf=0.25, iou=0.45)
results[0].save(filename="fire_smoke_result.jpg")
print(results)