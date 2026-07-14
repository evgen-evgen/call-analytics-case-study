from pydantic import BaseModel, Field


class AudioSource(BaseModel):
    file_id: str
    filename: str
    content_type: str


class DiarizationSegment(BaseModel):
    speaker: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class WordTimestamp(BaseModel):
    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class RawTranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[WordTimestamp]
