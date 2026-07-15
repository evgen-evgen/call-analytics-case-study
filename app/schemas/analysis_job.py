from enum import StrEnum

from pydantic import BaseModel

from app.schemas.analysis_response import AnalyzeResponse


class AnalysisJobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalyzeAccepted(BaseModel):
    job_id: str
    status: AnalysisJobStatus
    status_url: str


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: AnalysisJobStatus
    result: AnalyzeResponse | None = None
    error: str | None = None
