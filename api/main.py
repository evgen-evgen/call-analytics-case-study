from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field


app = FastAPI(
    title="MTBank Call Analysis API",
    version="0.1.0",
    description="ASR and multi-agent call analysis service.",
)


class TranscriptSegment(BaseModel):
    speaker: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class ClassificationResult(BaseModel):
    topic: str
    priority: str


class QualityChecklist(BaseModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool


class QualityResult(BaseModel):
    total: int = Field(ge=0, le=100)
    checklist: QualityChecklist


class ComplianceResult(BaseModel):
    passed: bool
    issues: list[dict]


class AnalysisResult(BaseModel):
    transcript: list[TranscriptSegment]
    classification: ClassificationResult
    quality_score: QualityResult
    compliance: ComplianceResult
    summary: str
    action_items: list[str]


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "analysis-api",
    }


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    message: Annotated[str | None, Form()] = None,
) -> AnalysisResult:
    """
    Milestone 1: returns a mock result.

    Later this endpoint will:
    1. resolve file or URL;
    2. run ASR;
    3. run diarization;
    4. execute analytical agents.
    """

    if file is not None and url is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide either file or url, not both.",
        )
    
 

    source_description = "OpenWebUI smoke test"

    if file is not None:
        source_description = f"Uploaded file: {file.filename}"
    elif url is not None:
        source_description = f"Audio URL: {url}"
    elif message:
        source_description = f"Message: {message}"

    return AnalysisResult(
        transcript=[],
        classification=ClassificationResult(
            topic="unknown",
            priority="medium",
        ),
        quality_score=QualityResult(
            total=0,
            checklist=QualityChecklist(
                greeting=False,
                need_detection=False,
                solution_provided=False,
                farewell=False,
            ),
        ),
        compliance=ComplianceResult(
            passed=True,
            issues=[],
        ),
        summary=(
            "Analysis service is connected successfully. "
            f"Request source: {source_description}."
        ),
        action_items=[],
    )