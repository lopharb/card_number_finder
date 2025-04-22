from services.card_detection import CardDetector
from services.ocr import OCRModel


detector = CardDetector("weights/yolo/yolov8m-seg-best.pt")
ocr = OCRModel(
    recognizer_path="weights/ocr/recognizer",
    detector_path="weights/ocr/detector",
    classifier_path="weights/ocr/angle_classifier"
)
