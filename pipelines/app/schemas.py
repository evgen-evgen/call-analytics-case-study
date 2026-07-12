from pydantic import BaseModel, Field


class AudioSource(BaseModel):
    file_id: str
    filename: str
    content_type: str

class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float =  Field(ge=0)
    text: str
    speaker: str

class DiariztionSegment(BaseModel):
    speaker: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class WordTImeStamp(BaseModel):
    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)