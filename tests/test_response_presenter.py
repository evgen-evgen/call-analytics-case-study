from app.agents.supervisor import AnalysisResult
from app.mappers import AnalyzeResponseMapper
from app.response_presenter import ResponsePresenter
from app.schemas import (
    ActionItem,
    ActionItemOwner,
    CallPriority,
    CallTopic,
    ClassificationResult,
    ComplianceAssessment,
    ComplianceCategory,
    ComplianceEvidence,
    ComplianceIssue,
    ComplianceResult,
    ComplianceSeverity,
    ComplianceStatus,
    DisclaimerCheck,
    QualityChecklist,
    QualityCriterionResult,
    QualityResult,
    RecommendationCorrectness,
    SummaryAssessment,
    TranscriptSegment,
)


def test_formats_transcript_as_json_and_agent_results_as_text() -> None:
    analysis = AnalysisResult(
        transcript=[
            TranscriptSegment(
                speaker="Клиент",
                start=1.2,
                end=3.4,
                text="Неизвестная операция по карте.",
            )
        ],
        classification=ClassificationResult(
            topic=CallTopic.CARDS,
            priority=CallPriority.HIGH,
            reasoning="Возможное мошенничество.",
        ),
        quality=QualityResult(
            checklist=QualityChecklist(
                greeting=QualityCriterionResult(passed=True, reason="Есть приветствие."),
                need_identification=QualityCriterionResult(passed=True, reason="Запрос уточнён."),
                solution=QualityCriterionResult(passed=True, reason="Решение дано."),
                farewell=QualityCriterionResult(passed=False, reason="Нет прощания."),
            ),
            passed_count=3,
            score=75,
        ),
        compliance=ComplianceResult(
            passed=False,
            risk_level=ComplianceSeverity.CRITICAL,
            issue_count=1,
            assessment=ComplianceAssessment(
                prohibited_phrases_found=True,
                disclaimer=DisclaimerCheck(
                    required=False,
                    present=None,
                    reason="Не требуется.",
                ),
                recommendation_correctness=RecommendationCorrectness(
                    status=ComplianceStatus.COMPLIANT,
                    reason="Рекомендация корректна.",
                ),
                issues=[
                    ComplianceIssue(
                        category=ComplianceCategory.UNSAFE_DATA_REQUEST,
                        severity=ComplianceSeverity.CRITICAL,
                        description="Запрошен PIN-код.",
                        recommendation="Не запрашивать секретные данные.",
                        evidence=ComplianceEvidence(
                            quote="Назовите PIN-код.",
                            speaker="Оператор",
                            start=5.0,
                            end=6.0,
                        ),
                    )
                ],
            ),
        ),
        summary=SummaryAssessment(
            summary=(
                "Клиент оспаривает операцию по карте. "
                "Оператор проверил обращение. "
                "Карту необходимо заблокировать."
            ),
            action_items=[
                ActionItem(
                    action="Заблокировать карту",
                    owner=ActionItemOwner.CLIENT,
                    reason="Предотвратить новые операции.",
                )
            ],
        ),
    )

    output = ResponsePresenter().format_analysis(
        AnalyzeResponseMapper().map(analysis),
        "<!-- marker -->",
    )

    assert "## Транскрипция" in output
    assert '"speaker": "Клиент"' in output
    assert "## Классификация" in output
    assert "**Тема:** карты" in output
    assert "**Оценка:** 75/100" in output
    assert "❌ Обнаружены нарушения" in output
    assert "## Комплаенс-проверка" in output
    assert "запрос секретных данных (критический риск)" in output
    assert "Цитата: «Назовите PIN-код.»" in output
    assert "Клиент оспаривает операцию по карте." in output
    assert "Заблокировать карту" in output
    assert '"classification"' not in output
    assert "unsafe_data_request" not in output
    assert "critical" not in output
    assert "## Compliance" not in output
