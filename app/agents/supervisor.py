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
    SummaryAssessment,
    TranscriptSegment,
)


from app.services import (
    calculate_quality_result,
    validate_quality_evidence,
    build_compliance_result,
    validate_compliance_evidence,
    normalize_summary_result,
)
class AnalysisResult(BaseModel):
    transcript: list[TranscriptSegment]
    classification: ClassificationResult
    quality: QualityResult
    compliance: ComplianceResult
    summary: SummaryAssessment

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
            asyncio.create_task(
                self.classifier.run(transcript),
                name="classification_agent",
            ),
            asyncio.create_task(
                self.quality.run(transcript),
                name="quality_agent",
            ),
            asyncio.create_task(
                self.compliance.run(transcript),
                name="compliance_agent",
            ),
            asyncio.create_task(
                self.summarizer.run(transcript),
                name="summarizer_agent",
            ),
        ]

        try:
            (
                classification,
                quality_assessment,
                compliance_assessment,
                summary_assessment,
            ) = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        quality_assessment = validate_quality_evidence(
            quality_assessment,
            transcript,
        )

        quality = calculate_quality_result(
            quality_assessment
        )

        compliance_assessment = (
            validate_compliance_evidence(
                compliance_assessment,
                transcript,
            )
        )

        compliance = build_compliance_result(
            compliance_assessment
        )
        summary = normalize_summary_result(
            summary_assessment
        )

        return AnalysisResult(
            transcript=transcript,
            classification=classification,
            quality=quality,
            compliance=compliance,
            summary=summary,
        )
