from fastapi import APIRouter, HTTPException, UploadFile, File
import cv2
import numpy as np

from .models import detector, ocr
from .schemas.card_number import CardNumberResponse
from .schemas.healthcheck import HealthCheckResponse


card_router = APIRouter(prefix='/api/v1')


@card_router.post("/get_card_number", response_model=CardNumberResponse, summary="Detect and extract card numbers")
async def get_card_number(file: UploadFile = File(...)):
    """
    Detect and extract card numbers from the uploaded image file.

    Args:
        file (UploadFile): The uploaded image file containing one or more card images.

    Returns:
        CardNumberResponse: A response model containing a list of detected card numbers.

    Raises:
        HTTPException: An HTTP exception with error details in case of an exception.
    """

    try:
        contents = await file.read()
        img_bytes = np.asarray(bytearray(contents), dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        card_images = detector.get_card_images(image)
        numbers = []

        for card_img in card_images:
            number = ocr.get_card_number(card_img)
            numbers.append(number)

        return {"card_numbers": numbers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@card_router.get("/healthcheck", response_model=HealthCheckResponse, summary="Health check")
def healthcheck():
    """
    A simple health check to ensure the API is running correctly.

    Returns a simple JSON response with a "status" key set to "ok".
    """
    return {"status": "ok"}
