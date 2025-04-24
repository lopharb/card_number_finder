from fastapi import APIRouter, HTTPException, UploadFile, File
import cv2
import numpy as np

from .models import detector, ocr
from .schemas.card_number import CardNumberResponse
from .schemas.healthcheck import HealthCheckResponse
from .logger import get_logger


card_router = APIRouter(prefix='/api/v1')
logger = get_logger()


@card_router.post("/get_card_number", response_model=CardNumberResponse, summary="Detect and extract card numbers")
async def get_card_number(file: UploadFile = File(...)):
    """
    Detect and extract card numbers from the uploaded image file.

    Args:
        file (UploadFile): The uploaded image file containing one or more card images.

    Returns:
        CardNumberResponse: A response model containing a list of detected card numbers.

    Raises:
        HTTPException: An HTTP exception with error details in case of an exception or if the request doesn't contain a file or the file is not an image.
    """

    try:
        if not file:
            raise HTTPException(status_code=400, detail="The request doesn't contain a file")

        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="The uploaded file is not an image")

        logger.info(f"Processing image from file {file.filename}")

        card_images = detector.get_card_images(image)
        numbers = []

        for card_img in card_images:
            number = ocr.get_card_number(card_img)
            numbers.append(number)

        logger.info(f"Extracted the following card numbers: {numbers}")

        return {"card_numbers": numbers}

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@card_router.get("/healthcheck", response_model=HealthCheckResponse, summary="Health check")
def healthcheck():
    """
    A simple health check to ensure the API is running correctly.

    Returns a simple JSON response with a "status" key set to "ok".
    """
    return {"status": "ok"}
