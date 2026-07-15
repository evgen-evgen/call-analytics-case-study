import pytest
from pydantic import ValidationError

from app.services.compliance_service import (
    build_compliance_result,
    validate_compliance_evidence,
)
from app.schemas import (
    ComplianceAssessment,
    ComplianceCategory,
    ComplianceEvidence,
    ComplianceIssue,
    ComplianceSeverity,
    ComplianceStatus,
    DisclaimerCheck,
    RecommendationCorrectness,
    TranscriptSegment,
)


def make_assessment(
    *,
    issues: list[ComplianceIssue] | None = None,
) -> ComplianceAssessment:
    return ComplianceAssessment(
        prohibited_phrases_found=bool(issues),
        disclaimer=DisclaimerCheck(
            required=False,
            present=None,
            reason="Disclaimer is not required.",
            evidence=None,
        ),
        recommendation_correctness=(
            RecommendationCorrectness(
                status=ComplianceStatus.COMPLIANT,
                reason="Recommendation is safe.",
                evidence=None,
            )
        ),
        issues=issues or [],
    )


def make_issue(
    *,
    severity: ComplianceSeverity,
    evidence: ComplianceEvidence | None = None,
) -> ComplianceIssue:
    return ComplianceIssue(
        category=(
            ComplianceCategory.PROHIBITED_PHRASE
        ),
        severity=severity,
        description="Найдена запрещённая фраза.",
        recommendation=(
            "Использовать нейтральную формулировку."
        ),
        evidence=evidence,
    )


def test_build_compliance_result_passes_without_issues() -> None:
    assessment = make_assessment()

    result = build_compliance_result(
        assessment
    )

    assert result.passed is True
    assert result.issue_count == 0
    assert result.risk_level == ComplianceSeverity.LOW


@pytest.mark.parametrize(
    ("severity", "expected_risk"),
    [
        (
            ComplianceSeverity.LOW,
            ComplianceSeverity.LOW,
        ),
        (
            ComplianceSeverity.MEDIUM,
            ComplianceSeverity.MEDIUM,
        ),
        (
            ComplianceSeverity.HIGH,
            ComplianceSeverity.HIGH,
        ),
        (
            ComplianceSeverity.CRITICAL,
            ComplianceSeverity.CRITICAL,
        ),
    ],
)
def test_build_compliance_result_uses_issue_severity(
    severity: ComplianceSeverity,
    expected_risk: ComplianceSeverity,
) -> None:
    assessment = make_assessment(
        issues=[
            make_issue(
                severity=severity
            )
        ]
    )

    result = build_compliance_result(
        assessment
    )

    assert result.passed is False
    assert result.issue_count == 1
    assert result.risk_level == expected_risk


def test_build_compliance_result_uses_highest_severity() -> None:
    assessment = make_assessment(
        issues=[
            make_issue(
                severity=ComplianceSeverity.LOW
            ),
            make_issue(
                severity=ComplianceSeverity.CRITICAL
            ),
            make_issue(
                severity=ComplianceSeverity.MEDIUM
            ),
        ]
    )

    result = build_compliance_result(
        assessment
    )

    assert result.issue_count == 3
    assert (
        result.risk_level
        == ComplianceSeverity.CRITICAL
    )


def test_compliance_evidence_rejects_invalid_interval() -> None:
    with pytest.raises(
        ValidationError,
        match="Evidence end must not be earlier",
    ):
        ComplianceEvidence(
            quote="Назовите PIN-код",
            speaker="Оператор",
            start=5.0,
            end=2.0,
        )


def test_validate_compliance_evidence_keeps_valid_quote() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text="Назовите PIN-код вашей карты.",
        ),
    ]

    assessment = make_assessment(
        issues=[
            make_issue(
                severity=ComplianceSeverity.CRITICAL,
                evidence=ComplianceEvidence(
                    quote="Назовите PIN-код",
                    speaker="Оператор",
                    start=0.0,
                    end=2.0,
                ),
            )
        ]
    )

    validated = validate_compliance_evidence(
        assessment,
        transcript,
    )

    assert validated.issues[0].evidence is not None


def test_validate_compliance_evidence_removes_hallucinated_quote() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text="Назовите последние четыре цифры карты.",
        ),
    ]

    assessment = make_assessment(
        issues=[
            make_issue(
                severity=ComplianceSeverity.CRITICAL,
                evidence=ComplianceEvidence(
                    quote="Назовите PIN-код",
                    speaker="Оператор",
                    start=0.0,
                    end=2.0,
                ),
            )
        ]
    )

    validated = validate_compliance_evidence(
        assessment,
        transcript,
    )

    assert validated.issues[0].evidence is None
    assert (
        "не прошла проверку"
        in validated.issues[0].description
    )


def test_validate_compliance_evidence_checks_speaker() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=2.0,
            text="Я могу сообщить PIN-код.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=2.2,
            end=4.0,
            text="Не сообщайте PIN-код.",
        ),
    ]

    assessment = make_assessment(
        issues=[
            make_issue(
                severity=ComplianceSeverity.CRITICAL,
                evidence=ComplianceEvidence(
                    quote="Я могу сообщить PIN-код",
                    speaker="Оператор",
                    start=0.0,
                    end=2.0,
                ),
            )
        ]
    )

    validated = validate_compliance_evidence(
        assessment,
        transcript,
    )

    assert validated.issues[0].evidence is None


def test_not_verifiable_does_not_create_failure() -> None:
    assessment = ComplianceAssessment(
        prohibited_phrases_found=False,
        disclaimer=DisclaimerCheck(
            required=False,
            present=None,
            reason="Disclaimer is not required.",
            evidence=None,
        ),
        recommendation_correctness=(
            RecommendationCorrectness(
                status=(
                    ComplianceStatus.NOT_VERIFIABLE
                ),
                reason=(
                    "Ставку нельзя проверить "
                    "без базы знаний."
                ),
                evidence=None,
            )
        ),
        issues=[],
    )

    result = build_compliance_result(
        assessment
    )

    assert result.passed is True
    assert result.issue_count == 0
