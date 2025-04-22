from api.endpoints import card_router
from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="Card Number Recognition API",
    description="API for detecting and extracting card numbers from images.",
    version="1.0.0"
)

app.include_router(card_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
