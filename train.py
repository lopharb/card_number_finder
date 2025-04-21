from ultralytics import YOLO
import mlflow
from training.mlflow_trainer import MLflowSegmentationTrainer

model = YOLO("weights/yolo/yolov8n-seg.pt")

# Optional: set experiment manually
mlflow.set_experiment("Card Detection Segmentation")


model.train(
    trainer=MLflowSegmentationTrainer,
    data='data/credit_card_no_augment_instance_seg/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    save=True,
    verbose=True
)
