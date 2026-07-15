import pytest
from pydantic import ValidationError

from app.services.quality_service import (
    calculate_quality_result,
    validate_quality_evidence,
)
from app.schemas import (
    QualityAssessment,
    QualityChecklist,
    QualityCriterionResult,
    QualityEvidence,
    TranscriptSegment,
)


def make_criterion(
    *,
    passed: bool,
    reason: str = "Test reason",
    evidence: QualityEvidence | None = None,
) -> QualityCriterionResult:
    return QualityCriterionResult(
        passed=passed,
        reason=reason,
        evidence=evidence,
    )


def make_assessment(
    *,
    greeting: bool,
    need_identification: bool,
    solution: bool,
    farewell: bool,
) -> QualityAssessment:
    return QualityAssessment(
        checklist=QualityChecklist(
            greeting=make_criterion(
                passed=greeting
            ),
            need_identification=make_criterion(
                passed=need_identification
            ),
            solution=make_criterion(
                passed=solution
            ),
            farewell=make_criterion(
                passed=farewell
            ),
        )
    )


@pytest.mark.parametrize(
    (
        "greeting",
        "need_identification",
        "solution",
        "farewell",
        "expected_passed_count",
        "expected_score",
    ),
    [
        (True, True, True, True, 4, 100),
        (True, True, True, False, 3, 75),
        (True, False, False, False, 1, 25),
        (False, False, False, False, 0, 0),
        (False, True, True, False, 2, 50),
    ],
)
def test_calculate_quality_result(
    greeting: bool,
    need_identification: bool,
    solution: bool,
    farewell: bool,
    expected_passed_count: int,
    expected_score: int,
) -> None:
    assessment = make_assessment(
        greeting=greeting,
        need_identification=need_identification,
        solution=solution,
        farewell=farewell,
    )

    result = calculate_quality_result(
        assessment
    )

    assert result.passed_count == expected_passed_count
    assert result.score == expected_score
    assert result.checklist == assessment.checklist


def test_quality_evidence_rejects_invalid_interval() -> None:
    with pytest.raises(
        ValidationError,
        match="Evidence end must not be earlier",
    ):
        QualityEvidence(
            quote="Добрый день",
            start=4.0,
            end=2.0,
        )


def test_validate_quality_evidence_keeps_valid_quote() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text="Добрый день, чем могу помочь?",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=3.2,
            end=6.0,
            text="Хочу узнать про кредит.",
        ),
    ]

    assessment = QualityAssessment(
        checklist=QualityChecklist(
            greeting=make_criterion(
                passed=True,
                reason="Оператор поздоровался.",
                evidence=QualityEvidence(
                    quote="Добрый день",
                    start=0.0,
                    end=1.0,
                ),
            ),
            need_identification=make_criterion(
                passed=False
            ),
            solution=make_criterion(
                passed=False
            ),
            farewell=make_criterion(
                passed=False
            ),
        )
    )

    validated = validate_quality_evidence(
        assessment,
        transcript,
    )

    assert (
        validated.checklist.greeting.evidence
        is not None
    )
    assert (
        validated.checklist.greeting.evidence.quote
        == "Добрый день"
    )


def test_validate_quality_evidence_removes_client_quote() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=2.0,
            text="Слушаю вас.",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=2.1,
            end=5.0,
            text="Добрый день, у меня вопрос.",
        ),
    ]

    assessment = QualityAssessment(
        checklist=QualityChecklist(
            greeting=make_criterion(
                passed=True,
                reason="Было приветствие.",
                evidence=QualityEvidence(
                    quote="Добрый день",
                    start=2.1,
                    end=3.0,
                ),
            ),
            need_identification=make_criterion(
                passed=False
            ),
            solution=make_criterion(
                passed=False
            ),
            farewell=make_criterion(
                passed=False
            ),
        )
    )

    validated = validate_quality_evidence(
        assessment,
        transcript,
    )

    assert (
        validated.checklist.greeting.evidence
        is None
    )
    assert (
        "не прошла проверку"
        in validated.checklist.greeting.reason
    )


def test_validate_quality_evidence_removes_hallucinated_quote() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=4.0,
            text="Добрый день, чем могу помочь?",
        ),
    ]

    assessment = QualityAssessment(
        checklist=QualityChecklist(
            greeting=make_criterion(
                passed=True,
                evidence=QualityEvidence(
                    quote="Здравствуйте, меня зовут Анна",
                    start=0.0,
                    end=2.0,
                ),
            ),
            need_identification=make_criterion(
                passed=False
            ),
            solution=make_criterion(
                passed=False
            ),
            farewell=make_criterion(
                passed=False
            ),
        )
    )

    validated = validate_quality_evidence(
        assessment,
        transcript,
    )

    assert (
        validated.checklist.greeting.evidence
        is None
    )


def test_validate_quality_evidence_removes_wrong_timestamp() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text="Добрый день.",
        ),
    ]

    assessment = QualityAssessment(
        checklist=QualityChecklist(
            greeting=make_criterion(
                passed=True,
                evidence=QualityEvidence(
                    quote="Добрый день",
                    start=15.0,
                    end=16.0,
                ),
            ),
            need_identification=make_criterion(
                passed=False
            ),
            solution=make_criterion(
                passed=False
            ),
            farewell=make_criterion(
                passed=False
            ),
        )
    )

    validated = validate_quality_evidence(
        assessment,
        transcript,
    )

    assert (
        validated.checklist.greeting.evidence
        is None
    )
