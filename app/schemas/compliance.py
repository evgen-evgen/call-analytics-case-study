from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ComplianceStatus(StrEnum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    NOT_VERIFIABLE = "not_verifiable"


class ComplianceCategory(StrEnum):
    PROHIBITED_PHRASE = "prohibited_phrase"
    MISSING_DISCLAIMER = "missing_disclaimer"
    INCORRECT_RECOMMENDATION = "incorrect_recommendation"
    UNSAFE_DATA_REQUEST = "unsafe_data_request"
    MISLEADING_PROMISE = "misleading_promise"
    OTHER = "other"


class ComplianceSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceEvidence(BaseModel):
    quote: str = Field(
        min_length=1,
        max_length=500,
    )
    speaker: str = Field(
        min_length=1,
        max_length=100,
    )
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "ComplianceEvidence":
        if self.end < self.start:
            raise ValueError(
                "Evidence end must not be earlier than start."
            )

        return self


class ComplianceIssue(BaseModel):
    category: ComplianceCategory
    severity: ComplianceSeverity

    description: str = Field(
        min_length=1,
        max_length=700,
        description="Описание нарушения только на русском языке.",
    )

    recommendation: str = Field(
        min_length=1,
        max_length=700,
        description="Рекомендация оператору только на русском языке.",
    )

    evidence: ComplianceEvidence | None = None


class DisclaimerCheck(BaseModel):
    required: bool

    present: bool | None = Field(
        description=(
            "True or false when disclaimer is required. "
            "Null when disclaimer is not applicable."
        )
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
        description="Причина результата только на русском языке.",
    )

    evidence: ComplianceEvidence | None = None


class RecommendationCorrectness(BaseModel):
    status: ComplianceStatus

    reason: str = Field(
        min_length=1,
        max_length=700,
        description="Обоснование результата только на русском языке.",
    )

    evidence: ComplianceEvidence | None = None


class ComplianceAssessment(BaseModel):
    prohibited_phrases_found: bool

    disclaimer: DisclaimerCheck

    recommendation_correctness: RecommendationCorrectness

    issues: list[ComplianceIssue] = Field(
        default_factory=list,
        max_length=5,
    )


class ComplianceResult(BaseModel):
    passed: bool
    risk_level: ComplianceSeverity
    assessment: ComplianceAssessment
    issue_count: int = Field(ge=0)
