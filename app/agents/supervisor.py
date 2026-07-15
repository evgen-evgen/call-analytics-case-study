import asyncio
import logging
from typing import Any

from app.agents.classifier import ClassificationAgent
from app.agents.compliance import ComplianceAgent
from app.agents.quality import QualityAgent
from app.agents.summarizer import SummarizerAgent
from app.llm.client import LLMClient
from app.schemas import (
    CallAnalysisResult,
    TranscriptSegment,
)
from app.services.compliance_service import (
    build_compliance_result,
    validate_compliance_evidence,
)
from app.services.quality_service import (
    calculate_quality_result,
    validate_quality_evidence,
)
from app.services.summary_service import normalize_summary_result
from app.observability import log_event


AnalysisResult = CallAnalysisResult


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
    ) -> CallAnalysisResult:
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

        results = await asyncio.gather(*tasks, return_exceptions=True)
        names = ("classification", "quality", "compliance", "summary")
        values: dict[str, Any | None] = {}
        agent_errors: dict[str, str] = {}

        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                agent_errors[name] = type(result).__name__
                log_event(
                    "agent.failed",
                    level=logging.ERROR,
                    agent=name,
                    error_type=type(result).__name__,
                )
                values[name] = None
            else:
                values[name] = result

        classification = values["classification"]
        quality_assessment = values["quality"]
        compliance_assessment = values["compliance"]
        summary_assessment = values["summary"]

        quality = None
        if quality_assessment is not None:
            quality_assessment = validate_quality_evidence(
                quality_assessment,
                transcript,
            )
            quality = calculate_quality_result(quality_assessment)

        compliance = None
        if compliance_assessment is not None:
            compliance_assessment = validate_compliance_evidence(
                compliance_assessment,
                transcript,
            )
            compliance = build_compliance_result(compliance_assessment)

        summary = None
        if summary_assessment is not None:
            summary = normalize_summary_result(summary_assessment)

        return CallAnalysisResult(
            classification=classification,
            quality=quality,
            compliance=compliance,
            summary=summary,
            transcript=transcript,
            agent_errors=agent_errors,
        )
