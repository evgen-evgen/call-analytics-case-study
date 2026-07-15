from pydantic import BaseModel, Field

from app.schemas.classification import ClassificationResult
from app.schemas.compliance import ComplianceResult
from app.schemas.quality import QualityResult
from app.schemas.summary import SummaryAssessment
from app.schemas.transcript import TranscriptSegment


class CallAnalysisResult(BaseModel):
    transcript: list[TranscriptSegment]
    classification: ClassificationResult | None = None
    quality: QualityResult | None = None
    compliance: ComplianceResult | None = None
    summary: SummaryAssessment | None = None
    agent_errors: dict[str, str] = Field(default_factory=dict)
