import os

from ..services.ocr import OCRModel
from ..services.card_detection import CardDetector

DETECTOR_PATH = "app/weights/yolo/"
OCR_PATH = "app/weights/ocr/"

detector = CardDetector(os.path.join(DETECTOR_PATH, "yolov8m-seg-best.pt"))
ocr = OCRModel(
    recognizer_path=os.path.join(OCR_PATH, "recognizer"),
    detector_path=os.path.join(OCR_PATH, "detector"),
    classifier_path=os.path.join(OCR_PATH, "angle_classifier"),
)
