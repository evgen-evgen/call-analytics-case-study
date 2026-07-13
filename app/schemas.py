from pydantic import BaseModel, Field
from enum import StrEnum

class AudioSource(BaseModel):
    file_id: str
    filename: str
    content_type: str



class DiariztionSegment(BaseModel):
    speaker: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class WordTImeStamp(BaseModel):
    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class RawTranscriptSegment(BaseModel):
    # raw transcript result 
    start: float = Field(ge=0)
    end: float =  Field(ge=0)
    text: str
    words: list[WordTImeStamp]

class TranscriptSegment(BaseModel):
    #final transcript segment with speaker

    start: float = Field(ge=0)
    end: float =  Field(ge=0)
    text: str
    speaker: str


class Topic(StrEnum):
    CREDITS = "кредиты"
    CARDS = "карты"
    TRANSFERS = "переводы"
    COMPLAINTS = "жалобы"
    OTHER = "другое"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ClassificationResult(BaseModel):
    topic: Topic
    priority: Priority
    reasoning: str = Field(
        min_length=1,
        description="Краткое обоснование классификации.",
    )

