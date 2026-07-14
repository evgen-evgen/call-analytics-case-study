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
        tasks = [
            asyncio.create_task(self.classifier.run(transcript)),
            asyncio.create_task(self.quality.run(transcript)),
            asyncio.create_task(self.compliance.run(transcript)),
            asyncio.create_task(self.summarizer.run(transcript)),
        ]

        try:
            (
                classification,
                quality,
                compliance,
                summary,
            ) = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return AnalysisResult(
            transcript=transcript,
            classification=classification,
            quality=quality,
            compliance=compliance,
            summary=summary,
        )
