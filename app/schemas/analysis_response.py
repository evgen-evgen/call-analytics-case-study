from pydantic import BaseModel, Field

from app.schemas.compliance import ComplianceIssue
from app.schemas.transcript import TranscriptSegment


class PublicClassification(BaseModel):
    topic: str
    priority: str


class PublicQualityChecklist(BaseModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool


class PublicQualityScore(BaseModel):
    total: int = Field(ge=0, le=100)
    checklist: PublicQualityChecklist


class PublicComplianceResult(BaseModel):
    passed: bool
    issues: list[ComplianceIssue]


class AnalyzeResponse(BaseModel):
    transcript: list[TranscriptSegment]
    classification: PublicClassification | None = None
    quality_score: PublicQualityScore | None = None
    compliance: PublicComplianceResult | None = None
    summary: str | None = None
    action_items: list[str] = Field(default_factory=list)
    agent_errors: dict[str, str] = Field(default_factory=dict)
