import asyncio

from app.agents.supervisor import AnalysisSupervisor
from app.schemas import (
    ActionItem,
    ActionItemOwner,
    CallPriority,
    CallTopic,
    ClassificationResult,
    ComplianceAssessment,
    ComplianceStatus,
    DisclaimerCheck,
    QualityAssessment,
    QualityChecklist,
    QualityCriterionResult,
    QualityResult,
    RecommendationCorrectness,
    SummaryAssessment,
    TranscriptSegment,
)


class FakeLLMClient:
    def __init__(self) -> None:
        self.active_requests = 0
        self.max_active_requests = 0

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
        temperature: float,
        response_format_mode: str = "json_schema",
    ):
        self.active_requests += 1
        self.max_active_requests = max(
            self.max_active_requests,
            self.active_requests,
        )

        await asyncio.sleep(0.01)
        self.active_requests -= 1

        if response_model is ClassificationResult:
            return ClassificationResult(
                topic=CallTopic.CARDS,
                priority=CallPriority.HIGH,
                reasoning="Неизвестная операция по карте.",
            )
        if response_model is QualityAssessment:
            return QualityAssessment(
                checklist=QualityChecklist(
                    greeting=QualityCriterionResult(passed=True, reason="Есть приветствие."),
                    need_identification=QualityCriterionResult(passed=True, reason="Запрос уточнён."),
                    solution=QualityCriterionResult(passed=True, reason="Решение дано."),
                    farewell=QualityCriterionResult(passed=False, reason="Нет прощания."),
                ),
            )
        if response_model is ComplianceAssessment:
            return ComplianceAssessment(
                prohibited_phrases_found=False,
                disclaimer=DisclaimerCheck(
                    required=False,
                    present=None,
                    reason="Не требуется.",
                ),
                recommendation_correctness=RecommendationCorrectness(
                    status=ComplianceStatus.COMPLIANT,
                    reason="Нарушений нет.",
                ),
            )
        if response_model is SummaryAssessment:
            return SummaryAssessment(
                summary=(
                    "Клиент сообщил о неизвестной операции. "
                    "Оператор принял запрос. "
                    "Карту нужно заблокировать."
                ),
                action_items=[
                    ActionItem(
                        action="Заблокировать карту.",
                        owner=ActionItemOwner.CLIENT,
                        reason="Исключить новые операции.",
                    )
                ],
            )
        raise AssertionError(f"Unexpected response model: {response_model}")


def test_runs_all_agents_concurrently() -> None:
    client = FakeLLMClient()
    supervisor = AnalysisSupervisor(client)
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=3.0,
            text="Я не совершал эту операцию по карте.",
        )
    ]

    result = asyncio.run(supervisor.run(transcript))

    assert client.max_active_requests == 4
    assert result.classification.priority == CallPriority.HIGH
    assert result.quality.score == 75
    assert result.compliance.passed is True
    assert result.summary.action_items[0].action == "Заблокировать карту."
