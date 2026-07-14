from app.schemas import (
    ComplianceAssessment,
    ComplianceEvidence,
    ComplianceResult,
    ComplianceSeverity,
    TranscriptSegment,
)


SEVERITY_ORDER = {
    ComplianceSeverity.LOW: 1,
    ComplianceSeverity.MEDIUM: 2,
    ComplianceSeverity.HIGH: 3,
    ComplianceSeverity.CRITICAL: 4,
}


def build_compliance_result(
    assessment: ComplianceAssessment,
) -> ComplianceResult:
    issue_count = len(assessment.issues)

    if not assessment.issues:
        risk_level = ComplianceSeverity.LOW
    else:
        risk_level = max(
            (
                issue.severity
                for issue in assessment.issues
            ),
            key=lambda severity: SEVERITY_ORDER[
                severity
            ],
        )

    return ComplianceResult(
        passed=issue_count == 0,
        risk_level=risk_level,
        assessment=assessment,
        issue_count=issue_count,
    )


def validate_compliance_evidence(
    assessment: ComplianceAssessment,
    transcript: list[TranscriptSegment],
) -> ComplianceAssessment:
    transcript_data = assessment.model_dump()

    for issue in transcript_data["issues"]:
        evidence_data = issue.get("evidence")

        if evidence_data is None:
            continue

        evidence = ComplianceEvidence.model_validate(
            evidence_data
        )

        if not _evidence_matches_transcript(
            evidence=evidence,
            transcript=transcript,
        ):
            issue["evidence"] = None
            issue["description"] = (
                f"{issue['description']} "
                "Указанная цитата не прошла "
                "проверку по транскрипту."
            )

    disclaimer_evidence = transcript_data[
        "disclaimer"
    ].get("evidence")

    if disclaimer_evidence is not None:
        evidence = ComplianceEvidence.model_validate(
            disclaimer_evidence
        )

        if not _evidence_matches_transcript(
            evidence=evidence,
            transcript=transcript,
        ):
            transcript_data["disclaimer"][
                "evidence"
            ] = None

    correctness_evidence = transcript_data[
        "recommendation_correctness"
    ].get("evidence")

    if correctness_evidence is not None:
        evidence = ComplianceEvidence.model_validate(
            correctness_evidence
        )

        if not _evidence_matches_transcript(
            evidence=evidence,
            transcript=transcript,
        ):
            transcript_data[
                "recommendation_correctness"
            ]["evidence"] = None

    return ComplianceAssessment.model_validate(
        transcript_data
    )


def _evidence_matches_transcript(
    *,
    evidence: ComplianceEvidence,
    transcript: list[TranscriptSegment],
    timestamp_tolerance: float = 0.5,
) -> bool:
    normalized_quote = _normalize_text(
        evidence.quote
    )

    for segment in transcript:
        if (
            _normalize_text(segment.speaker)
            != _normalize_text(evidence.speaker)
        ):
            continue

        quote_matches = (
            normalized_quote
            and normalized_quote
            in _normalize_text(segment.text)
        )

        timestamps_match = (
            evidence.start
            >= segment.start - timestamp_tolerance
            and evidence.end
            <= segment.end + timestamp_tolerance
        )

        if quote_matches and timestamps_match:
            return True

    return False


def _normalize_text(value: str) -> str:
    return " ".join(
        value.lower().strip().split()
    )