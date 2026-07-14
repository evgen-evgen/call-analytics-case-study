from pydantic import BaseModel, Field
from enum import StrEnum

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
    # raw transcript result 
    start: float = Field(ge=0)
    end: float =  Field(ge=0)
    text: str
    words: list[WordTimestamp]

class TranscriptSegment(BaseModel):
    #final transcript segment with speaker

    start: float = Field(ge=0)
    end: float =  Field(ge=0)
    text: str
    speaker: str




class CallTopic(StrEnum):
    CREDITS = "кредиты"
    CARDS = "карты"
    TRANSFERS = "переводы"
    COMPLAINTS = "жалобы"
    OTHER = "другое"


class CallPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClassificationResult(BaseModel):
    topic: CallTopic
    priority: CallPriority

    reasoning: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Краткое объяснение классификации "
            "на основании транскрипта."
        ),
    )

class QualityChecklist(BaseModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool


class QualityResult(BaseModel):
    total: int = Field(ge=0, le=100)
    checklist: QualityChecklist
    issues: list[str] = Field(default_factory=list)


class ComplianceIssue(BaseModel):
    category: str
    description: str
    quote: str | None = None
    start: float | None = None
    end: float | None = None


class ComplianceResult(BaseModel):
    passed: bool
    issues: list[ComplianceIssue]


class SummaryResult(BaseModel):
    summary: str
    action_items: list[str]


class CallAnalysisResult(BaseModel):
    classification: ClassificationResult
    quality: QualityResult
    compliance: ComplianceResult
    summary: SummaryResult
