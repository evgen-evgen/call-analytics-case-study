from app.schemas import (
    QualityAssessment,
    QualityChecklist,
    QualityCriterionResult,
    QualityEvidence,
    QualityResult,
    TranscriptSegment,
)


QUALITY_CRITERIA = (
    "greeting",
    "need_identification",
    "solution",
    "farewell",
)

POINTS_PER_CRITERION = 25


def calculate_quality_result(
    assessment: QualityAssessment,
) -> QualityResult:
    checklist = assessment.checklist

    criteria = [
        checklist.greeting,
        checklist.need_identification,
        checklist.solution,
        checklist.farewell,
    ]

    passed_count = sum(
        criterion.passed
        for criterion in criteria
    )

    return QualityResult(
        checklist=checklist,
        passed_count=passed_count,
        score=passed_count * POINTS_PER_CRITERION,
    )


def validate_quality_evidence(
    assessment: QualityAssessment,
    transcript: list[TranscriptSegment],
) -> QualityAssessment:
    """
    Checks that evidence belongs to an operator segment and
    that the quote and timestamps match that segment.

    Invalid evidence is removed. The criterion itself is not
    automatically changed because the LLM may have used several
    operator phrases in its reasoning.
    """

    operator_segments = [
        segment
        for segment in transcript
        if segment.speaker == "Оператор"
    ]

    checklist_data = assessment.checklist.model_dump()

    for criterion_name in QUALITY_CRITERIA:
        criterion_data = checklist_data[criterion_name]
        evidence_data = criterion_data.get("evidence")

        if evidence_data is None:
            continue

        if not _evidence_matches_operator_segment(
            evidence=QualityEvidence.model_validate(
                evidence_data
            ),
            operator_segments=operator_segments,
        ):
            criterion_data["evidence"] = None
            criterion_data["reason"] = (
                f"{criterion_data['reason']} "
                "Указанная цитата не прошла проверку "
                "по транскрипту."
            )

    return QualityAssessment(
        checklist=QualityChecklist.model_validate(
            checklist_data
        )
    )


def _evidence_matches_operator_segment(
    *,
    evidence: QualityEvidence,
    operator_segments: list[TranscriptSegment],
    timestamp_tolerance: float = 0.5,
) -> bool:
    normalized_quote = _normalize_text(
        evidence.quote
    )

    for segment in operator_segments:
        normalized_segment = _normalize_text(
            segment.text
        )

        quote_matches = (
            normalized_quote
            and normalized_quote in normalized_segment
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