import cv2
import numpy as np
from ultralytics import YOLO


class CardDetector:
    def __init__(self, yolo_model_path: str):
        """
        Initialize the CardDetector with a YOLO model for card detection.

        Args:
            yolo_model_path (str): Path to the pre-trained YOLO model file used for detecting card images.
        """

        self.model = YOLO(yolo_model_path)

    def _align_card(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Aligns a detected card in the given image to a standard rectangle landscape orientation and crops it to only fit the card.

        Args:
            image (np.ndarray): The input image containing the detected card.
            mask (np.ndarray): The segmentation mask of the detected card.

        Returns:
            np.ndarray: The aligned and croppedcard image.
        """

        width, height, box = self._get_dimensions(mask)

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        if warped.shape[0] > warped.shape[1]:  # height > width
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    def _get_dimensions(self, points) -> tuple[int, int, np.ndarray]:
        """
        Computes the dimensions and corner points of a minimal rotated rectangle that encloses the given polygon points.

        Args:
            points (list): A list of 2D NumPy arrays representing the polygon.

        Returns:
            tuple: A tuple containing the width, height and the corners of the rectangle.
        """

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))  # type: ignore
        height = int(max(np.linalg.norm(box[1] - box[2]), np.linalg.norm(box[3] - box[0])))  # type: ignore

        return width, height, np.array(box)

    def _get_masks(self, results) -> list[np.ndarray]:
        """
        Retrieves the masks from a list of OCR result objects.

        Args:
            results (list): A list of OCR result objects.

        Returns:
            list[np.ndarray]: A list of NumPy arrays representing the masks of the detected text regions.
        """

        if not results or results[0].masks is None:
            return []

        polygons = results[0].masks.xy

        return [polygon for polygon in polygons]

    def get_card_images(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Detects and aligns card images present in the input image using a YOLO model.

        This function processes the input image to find and extract card images using a pre-trained
        YOLO model. It aligns the detected cards to a standard landscape orientation by applying 
        perspective transformation and cropping.

        Args:
            image (np.ndarray): The input image containing one or more cards.

        Returns:
            list[np.ndarray]: A list of aligned and cropped card images extracted from the input image.
        """

        results = self.model(image, verbose=False)
        masks = self._get_masks(results)

        aligned_cards = []
        for mask in masks:
            try:
                aligned = self._align_card(image, mask)
                aligned_cards.append(aligned)
            except Exception as e:
                print(f"Failed to align card: {e}")

        return aligned_cards
