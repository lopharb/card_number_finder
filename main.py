from contextlib import asynccontextmanager

from app.api.endpoints import card_router
from fastapi import FastAPI
import uvicorn
from app.api.logger import get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API is starting up...")
    yield
    logger.info("API is shutting down...")

app = FastAPI(
    title="Card Number Recognition API",
    description="API for detecting and extracting card numbers from images.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(card_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=False)
