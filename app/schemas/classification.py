from enum import StrEnum

from pydantic import BaseModel, Field


class CallTopic(StrEnum):
    CREDITS = "кредиты"
    CARDS = "карты"
    TRANSFERS = "переводы"
    COMPLAINTS = "жалобы"


class CallPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClassificationResult(BaseModel):
    topic: CallTopic
    priority: CallPriority
    reasoning: str = Field(
        min_length=1,
        description=(
            "Краткое объяснение классификации на основании "
            "транскрипта, желательно до 300 символов."
        ),
    )
