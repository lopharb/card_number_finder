import re
import cv2
from matplotlib import pyplot as plt
import numpy as np
from paddleocr import PaddleOCR


# TODO: fix logging disabling (xd)

# disable logging whenever we import this file
# from paddleocr.ppocr.utils.logging import get_logger
# import logging
# logger = get_logger()
# logger.setLevel(logging.ERROR)


class OCRModel:
    def __init__(self, recognizer_path: str, detector_path: str, classifier_path: str, use_angle_classifier: bool = False):
        """
        Initialize the OCRModel with specified model paths and angle classifier option.

        Args:
            recognizer_path (str): Path to the OCR recognition model. Will be downloaded automatically if not present.
            detector_path (str): Path to the OCR detection model. Will be downloaded automatically if not present.
            classifier_path (str): Path to the OCR angle classifier model. Will be downloaded automatically if not present.
            use_angle_classifier (bool, optional): Whether to use the angle classifier. Defaults to False.
        """

        self.use_classifier = use_angle_classifier
        self.ocr = PaddleOCR(use_angle_cls=self.use_classifier,
                             lang="en",
                             rec_model_dir=recognizer_path,
                             det_model_dir=detector_path,
                             cls_model_dir=classifier_path)

    def _are_close(self, box1: list, box2: list, x_thresh: int = 30, y_thresh: int = 20) -> bool:
        """
        Check if two boxes are close enough to be considered as one.

        Given two boxes, determine if they are close enough to be considered as one.
        The boxes are considered close if the x distance between the right edge of the first box
        and the left edge of the second box is less than x_thresh, and the y distance between the
        middle of the first box and the middle of the second box is less than y_thresh.

        Args:
            box1 (list): A list of four points representing the first box.
            box2 (list): A list of four points representing the second box.
            x_thresh (int, optional): The maximum x distance between the boxes. Defaults to 30.
            y_thresh (int, optional): The maximum y distance between the boxes. Defaults to 20.

        Returns:
            bool: Whether the boxes are close enough to be considered as one.
        """
        x1_max = max(p[0] for p in box1)
        x2_min = min(p[0] for p in box2)

        y1_mid = sum(p[1] for p in box1) / 4
        y2_mid = sum(p[1] for p in box2) / 4

        return abs(x2_min - x1_max) < x_thresh and abs(y1_mid - y2_mid) < y_thresh

    def _get_overlap_length(self, s1: str, s2: str) -> int:
        """
        Get the length of the overlap between two strings.

        Given two strings, return the length of the overlap between the two strings.
        The overlap is defined as the longest common suffix of the first string and the
        longest common prefix of the second string.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            int: The length of the overlap between the two strings.
        """
        l = 1
        overlap_length = 0
        while l <= min(len(s1), len(s2)):
            if s1[-l:] == s2[:l]:
                overlap_length = l
            l += 1
        return overlap_length

    def _merge_no_overlap(self, lines: list[str]) -> str:
        """
        Merge multiple strings into one by overlapping the longest common suffix of the current result
        and the longest common prefix of the next string.

        Args:
            lines (list[str]): The strings to merge.

        Returns:
            str: The merged string.
        """
        result = lines[0]
        for line in lines[1:]:
            result += line[self._get_overlap_length(result, line):]

        return result

    def _match_detections(self, ocr_lines: list) -> list[str]:
        """
        Try to match the OCR results to a card number.

        The function takes the OCR results and tries to merge the strings into a single card number.
        It works by grouping the strings by their y-coordinate and then by their x-coordinate.
        Then it tries to merge the strings in each group into a single string using the _merge_no_overlap
        function. The merged strings are then checked to see if they are a valid card number (16 digits).

        Args:
            ocr_lines (list): The OCR results.

        Returns:
            list[str]: A list of valid card numbers found in the OCR results.
        """
        filtered = [(re.sub(r'\s+', '', text), box)
                    for box, (text, _score) in ocr_lines if text.isnumeric() and len(text) <= 16]

        # FIXME we should sort by y and then by x
        sorted_blocks = sorted(filtered, key=lambda x: min(p[0] for p in x[1]))

        groups = []
        current_group = []

        for text, box in sorted_blocks:
            if not current_group:
                current_group.append((text, box))
                continue
            _, prev_box = current_group[-1]
            if self._are_close(prev_box, box):
                current_group.append((text, box))
            else:
                groups.append(current_group)
                current_group = [(text, box)]
        if current_group:
            groups.append(current_group)

        merged = []
        for group in groups:
            number = self._merge_no_overlap([text for text, _box in group])
            if len(number) == 16 and number.isdigit():
                merged.append(number)
        return merged\


    def _try_get_card_number(self, image: np.ndarray) -> str:
        """
        Tries to extract a card number from the given image by performing OCR with the model.

        Args:
            image (np.ndarray): The image to perform OCR on.

        Returns:
            str: The extracted card number or an empty string if none is found.
        """
        result = self.ocr.ocr(image, cls=self.use_classifier)

        if result[0] is None:
            return ''

        for line in result[0]:
            _, (text, _) = line
            text = re.sub(r'\s+', '', text)
            if text.isnumeric() and len(text) == 16:
                return text

        merged_numbers = self._match_detections(result[0])
        if merged_numbers:
            return merged_numbers[0]

        return ''

    def get_card_number(self, image: np.ndarray) -> str:
        """
        Perform OCR and try to extract the card number from the image.

        Args:
            image (np.ndarray): The image to perform OCR on.

        Returns:
            dict[str, str]: A dictionary containing the card number and its confidence.
        """
        card_number = self._try_get_card_number(image)
        if card_number != '':
            return card_number

        # handle an upside down card
        flipped_image = cv2.flip(image, -1)
        return self._try_get_card_number(flipped_image)
