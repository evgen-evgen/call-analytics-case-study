import asyncio
from time import perf_counter

import pytest

from app.agents.supervisor import AnalysisSupervisor
from app.schemas import (
    ClassificationResult,
    ComplianceAssessment,
    ComplianceStatus,
    DisclaimerCheck,
    QualityAssessment,
    QualityChecklist,
    QualityCriterionResult,
    RecommendationCorrectness,
    SummaryAssessment,
    TranscriptSegment,
)


class FakeAgent:
    def __init__(
        self,
        result=None,
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.delay = delay
        self.error = error

        self.calls = 0
        self.cancelled = False

    async def run(
        self,
        transcript: list[TranscriptSegment],
    ):
        self.calls += 1

        try:
            if self.delay:
                await asyncio.sleep(self.delay)

            if self.error is not None:
                raise self.error

            return self.result

        except asyncio.CancelledError:
            self.cancelled = True
            raise


def make_quality_criterion(
    passed: bool,
) -> QualityCriterionResult:
    return QualityCriterionResult(
        passed=passed,
        reason="Test reason",
        evidence=None,
    )


def make_quality_assessment() -> QualityAssessment:
    return QualityAssessment(
        checklist=QualityChecklist(
            greeting=make_quality_criterion(True),
            need_identification=make_quality_criterion(True),
            solution=make_quality_criterion(True),
            farewell=make_quality_criterion(False),
        )
    )


def make_compliance_assessment() -> ComplianceAssessment:
    return ComplianceAssessment(
        prohibited_phrases_found=False,
        disclaimer=DisclaimerCheck(
            required=False,
            present=None,
            reason="Disclaimer is not required.",
            evidence=None,
        ),
        recommendation_correctness=RecommendationCorrectness(
            status=ComplianceStatus.COMPLIANT,
            reason="Recommendation is safe.",
            evidence=None,
        ),
        issues=[],
    )


def make_summary_assessment() -> SummaryAssessment:
    return SummaryAssessment(
        summary=(
            "Клиент сообщил о задержанном переводе. "
            "Оператор уточнил детали операции. "
            "Обращение было зарегистрировано. "
            "Результат проверки пока ожидается."
        ),
        action_items=[],
    )


def make_classification_result() -> ClassificationResult:
    return ClassificationResult(
        topic="переводы",
        priority="medium",
        reasoning="Перевод не дошёл получателю.",
    )


def make_transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=2.0,
            text="Перевод не дошёл.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=2.2,
            end=5.0,
            text="Я зарегистрировал обращение.",
        ),
    ]


def make_supervisor(
    *,
    classifier: FakeAgent | None = None,
    quality: FakeAgent | None = None,
    compliance: FakeAgent | None = None,
    summarizer: FakeAgent | None = None,
) -> AnalysisSupervisor:
    """
    Создаём Supervisor без вызова __init__,
    чтобы не создавать настоящий LLMClient.
    """

    supervisor = AnalysisSupervisor.__new__(
        AnalysisSupervisor
    )

    supervisor.classifier = classifier or FakeAgent(
        make_classification_result()
    )

    supervisor.quality = quality or FakeAgent(
        make_quality_assessment()
    )

    supervisor.compliance = compliance or FakeAgent(
        make_compliance_assessment()
    )

    supervisor.summarizer = summarizer or FakeAgent(
        make_summary_assessment()
    )

    return supervisor


@pytest.mark.asyncio
async def test_supervisor_builds_analysis_result() -> None:
    supervisor = make_supervisor()
    transcript = make_transcript()

    result = await supervisor.run(transcript)

    assert result.transcript == transcript

    assert result.classification.topic == "переводы"
    assert result.classification.priority == "medium"

    assert result.quality.passed_count == 3
    assert result.quality.score == 75

    assert result.compliance.passed is True
    assert result.compliance.issue_count == 0

    assert (
        result.summary.summary
        == make_summary_assessment().summary
    )

    assert supervisor.classifier.calls == 1
    assert supervisor.quality.calls == 1
    assert supervisor.compliance.calls == 1
    assert supervisor.summarizer.calls == 1


@pytest.mark.asyncio
async def test_supervisor_runs_agents_concurrently() -> None:
    delay = 0.15

    supervisor = make_supervisor(
        classifier=FakeAgent(
            make_classification_result(),
            delay=delay,
        ),
        quality=FakeAgent(
            make_quality_assessment(),
            delay=delay,
        ),
        compliance=FakeAgent(
            make_compliance_assessment(),
            delay=delay,
        ),
        summarizer=FakeAgent(
            make_summary_assessment(),
            delay=delay,
        ),
    )

    started_at = perf_counter()

    await supervisor.run(
        make_transcript()
    )

    duration = perf_counter() - started_at

    # Последовательно было бы около 0.6 секунды.
    assert duration < 0.35


@pytest.mark.asyncio
async def test_supervisor_returns_other_results_when_agent_fails() -> None:
    supervisor = make_supervisor(
        classifier=FakeAgent(
            error=RuntimeError(
                "Classifier failed"
            )
        ),
        quality=FakeAgent(
            make_quality_assessment(),
            delay=1.0,
        ),
        compliance=FakeAgent(
            make_compliance_assessment(),
            delay=1.0,
        ),
        summarizer=FakeAgent(
            make_summary_assessment(),
            delay=1.0,
        ),
    )

    result = await supervisor.run(make_transcript())

    assert result.classification is None
    assert result.quality is not None
    assert result.compliance is not None
    assert result.summary is not None
    assert result.agent_errors == {"classification": "RuntimeError"}


@pytest.mark.asyncio
async def test_supervisor_does_not_cancel_other_agents_on_error() -> None:
    classifier = FakeAgent(
        error=RuntimeError(
            "Classifier failed"
        ),
        delay=0.01,
    )

    quality = FakeAgent(
        make_quality_assessment(),
        delay=0.02,
    )

    compliance = FakeAgent(
        make_compliance_assessment(),
        delay=0.02,
    )

    summarizer = FakeAgent(
        make_summary_assessment(),
        delay=0.02,
    )

    supervisor = make_supervisor(
        classifier=classifier,
        quality=quality,
        compliance=compliance,
        summarizer=summarizer,
    )

    result = await supervisor.run(make_transcript())

    assert result.classification is None
    assert quality.cancelled is False
    assert compliance.cancelled is False
    assert summarizer.cancelled is False


@pytest.mark.asyncio
async def test_supervisor_calls_every_agent_once() -> None:
    supervisor = make_supervisor()

    await supervisor.run(
        make_transcript()
    )

    assert supervisor.classifier.calls == 1
    assert supervisor.quality.calls == 1
    assert supervisor.compliance.calls == 1
    assert supervisor.summarizer.calls == 1


@pytest.mark.asyncio
async def test_supervisor_collects_successes_after_fast_failure() -> None:
    supervisor = AnalysisSupervisor.__new__(
        AnalysisSupervisor
    )

    classifier = FakeAgent(
        error=RuntimeError(
            "Classifier failed"
        ),
        delay=0.01,
    )

    quality = FakeAgent(
        make_quality_assessment(),
        delay=0.02,
    )

    compliance = FakeAgent(
        make_compliance_assessment(),
        delay=0.02,
    )

    summarizer = FakeAgent(
        make_summary_assessment(),
        delay=0.02,
    )

    supervisor.classifier = classifier
    supervisor.quality = quality
    supervisor.compliance = compliance
    supervisor.summarizer = summarizer

    result = await supervisor.run(make_transcript())

    assert result.classification is None
    assert result.quality is not None
    assert result.compliance is not None
    assert result.summary is not None
    assert result.agent_errors == {"classification": "RuntimeError"}
