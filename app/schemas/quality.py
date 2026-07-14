from pydantic import BaseModel, Field, model_validator


class QualityEvidence(BaseModel):
    quote: str = Field(min_length=1, max_length=500)
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "QualityEvidence":
        if self.end < self.start:
            raise ValueError(
                "Evidence end must not be earlier than start."
            )
        return self


class QualityCriterionResult(BaseModel):
    passed: bool
    reason: str = Field(min_length=1, max_length=500)
    evidence: QualityEvidence | None = None


class QualityChecklist(BaseModel):
    greeting: QualityCriterionResult
    need_identification: QualityCriterionResult
    solution: QualityCriterionResult
    farewell: QualityCriterionResult


class QualityAssessment(BaseModel):
    """Raw structured result returned by the LLM."""

    checklist: QualityChecklist


class QualityResult(BaseModel):
    """Final result after deterministic score calculation."""

    checklist: QualityChecklist
    passed_count: int = Field(ge=0, le=4)
    score: int = Field(ge=0, le=100)
