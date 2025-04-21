import os

from ultralytics import settings
from ultralytics import YOLO

from training.augmentations import AUGMENTATIONS
from training.mlflow_trainer import MLflowSegmentationTrainer


def train(model_path: str, data_path: str, augmentations: dict, **kwargs):
    """
    Train the YOLO model with the specified parameters.

    Args:
        model_path (str): Path to the YOLO model.
        data_path (str): Path to the data.
        augmentations (dict): Dictionary containing augmentation parameters.
        **kwargs: Additional keyword arguments to pass to the YOLO model.
    """
    settings.update({"mlflow": False})  # Disable built-in logging

    model = YOLO(model_path)

    # Get full path, otherwise the built-in handler breaks
    full_path = os.path.abspath(data_path)

    model.train(
        data=full_path,
        trainer=MLflowSegmentationTrainer,
        **kwargs,
        **augmentations
    )


if __name__ == "__main__":
    train(
        model_path='weights/yolo/yolov8m-seg-pretrained.pt',
        data_path='data/credit_card_no_augment_instance_seg/data.yaml',
        augmentations=AUGMENTATIONS,
        epochs=50,
        imgsz=640,
        batch=16,
        save=True,
        verbose=True,
        device=0,
        save_period=10,
    )
