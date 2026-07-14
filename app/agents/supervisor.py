import asyncio

from pydantic import BaseModel

from app.agents.classifier import ClassificationAgent
from app.agents.compliance import ComplianceAgent
from app.agents.quality import QualityAgent
from app.agents.summarizer import SummarizerAgent
from app.llm.client import LLMClient
from app.schemas import (
    ClassificationResult,
    ComplianceResult,
    QualityResult,
    SummaryResult,
    TranscriptSegment,
)


class AnalysisResult(BaseModel):
    transcript: list[TranscriptSegment]
    classification: ClassificationResult
    quality: QualityResult
    compliance: ComplianceResult
    summary: SummaryResult


class AnalysisSupervisor:
    def __init__(
        self,
        llm_client: LLMClient,
    ) -> None:
        self.classifier = ClassificationAgent(llm_client)
        self.quality = QualityAgent(llm_client)
        self.compliance = ComplianceAgent(llm_client)
        self.summarizer = SummarizerAgent(llm_client)

    async def run(
        self,
        transcript: list[TranscriptSegment],
    ) -> AnalysisResult:
        (
            classification,
            quality,
            compliance,
            summary,
        ) = await asyncio.gather(
            self.classifier.run(transcript),
            self.quality.run(transcript),
            self.compliance.run(transcript),
            self.summarizer.run(transcript),
        )

        return AnalysisResult(
            transcript=transcript,
            classification=classification,
            quality=quality,
            compliance=compliance,
            summary=summary,
        )
