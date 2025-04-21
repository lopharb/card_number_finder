import numpy as np
from ultralytics import YOLO


class DetectionModel:
    def __init__(self, yolo_model_path: str):
        self.model = YOLO(yolo_model_path)

    def _align_card(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def crop_card(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_card_image(self, image: np.ndarray) -> np.ndarray:
        results = self.model(image)
        # TODO implement
        raise NotImplementedError
