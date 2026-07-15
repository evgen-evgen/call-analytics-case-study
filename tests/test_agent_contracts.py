from typing import Generic, TypeVar

import pytest

from app.mappers import AnalyzeResponseMapper
from app.agents.classifier import ClassificationAgent
from app.agents.compliance import ComplianceAgent
from app.agents.quality import QualityAgent
from app.agents.summarizer import SummarizerAgent
from app.agents.supervisor import AnalysisSupervisor
from app.response_presenter import ResponsePresenter
from app.schemas import (
    CallAnalysisResult,
    CallPriority,
    CallTopic,
    ClassificationResult,
    ComplianceAssessment,
    ComplianceSeverity,
    ComplianceStatus,
    DisclaimerCheck,
    QualityAssessment,
    QualityChecklist,
    QualityCriterionResult,
    RecommendationCorrectness,
    SummaryAssessment,
    TranscriptSegment,
)


AgentResult = TypeVar("AgentResult")


class StubAgent(Generic[AgentResult]):
    def __init__(self, result: AgentResult) -> None:
        self.result = result

    async def run(
        self,
        transcript: list[TranscriptSegment],
    ) -> AgentResult:
        return self.result


def test_agents_expose_the_expected_raw_result_models() -> None:
    assert ClassificationAgent.result_model is ClassificationResult
    assert QualityAgent.result_model is QualityAssessment
    assert ComplianceAgent.result_model is ComplianceAssessment
    assert SummarizerAgent.result_model is SummaryAssessment


@pytest.mark.asyncio
async def test_all_agent_results_pass_through_supervisor_and_presenter() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0,
            end=4,
            text="Здравствуйте. Чем могу помочь?",
        )
    ]
    criterion = QualityCriterionResult(
        passed=True,
        reason="Критерий выполнен.",
    )

    supervisor = AnalysisSupervisor.__new__(AnalysisSupervisor)
    supervisor.classifier = StubAgent(
        ClassificationResult(
            topic=CallTopic.CARDS,
            priority=CallPriority.LOW,
            reasoning="Вопрос о банковской карте.",
        )
    )
    supervisor.quality = StubAgent(
        QualityAssessment(
            checklist=QualityChecklist(
                greeting=criterion,
                need_identification=criterion,
                solution=criterion,
                farewell=criterion,
            )
        )
    )
    supervisor.compliance = StubAgent(
        ComplianceAssessment(
            prohibited_phrases_found=False,
            disclaimer=DisclaimerCheck(
                required=False,
                present=None,
                reason="Дисклеймер не требуется.",
            ),
            recommendation_correctness=RecommendationCorrectness(
                status=ComplianceStatus.COMPLIANT,
                reason="Нарушений нет.",
            ),
        )
    )
    supervisor.summarizer = StubAgent(
        SummaryAssessment(
            summary=(
                "Клиент обратился в банк. "
                "Оператор принял запрос. "
                "Звонок завершён."
            )
        )
    )

    result = await supervisor.run(transcript)

    assert isinstance(result, CallAnalysisResult)
    assert result.quality.score == 100
    assert result.compliance.passed is True
    assert result.compliance.risk_level is ComplianceSeverity.LOW
    rendered = ResponsePresenter().format_analysis(
        AnalyzeResponseMapper().map(result),
        "<!-- test -->",
    )
    assert "## Классификация" in rendered
    assert "**Оценка:** 100/100" in rendered
    assert "Нарушений не обнаружено." in rendered
