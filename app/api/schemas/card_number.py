from pydantic import BaseModel
from typing import List


class CardNumberResponse(BaseModel):
    card_numbers: List[str]
