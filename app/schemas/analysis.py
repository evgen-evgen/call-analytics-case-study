from pydantic import BaseModel

from app.schemas.classification import ClassificationResult
from app.schemas.compliance import ComplianceResult
from app.schemas.quality import QualityResult
from app.schemas.summary import SummaryResult


class CallAnalysisResult(BaseModel):
    classification: ClassificationResult
    quality: QualityResult
    compliance: ComplianceResult
    summary: SummaryResult
