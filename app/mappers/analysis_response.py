from app.schemas import (
    AnalyzeResponse,
    CallAnalysisResult,
    PublicClassification,
    PublicComplianceResult,
    PublicQualityChecklist,
    PublicQualityScore,
)


class AnalyzeResponseMapper:
    """Maps the internal analysis model to the shared public contract."""

    def map(self, result: CallAnalysisResult) -> AnalyzeResponse:
        classification = None
        if result.classification is not None:
            classification = PublicClassification(
                topic=result.classification.topic.value,
                priority=result.classification.priority.value,
            )

        quality_score = None
        if result.quality is not None:
            checklist = result.quality.checklist
            quality_score = PublicQualityScore(
                total=result.quality.score,
                checklist=PublicQualityChecklist(
                    greeting=checklist.greeting.passed,
                    need_detection=checklist.need_identification.passed,
                    solution_provided=checklist.solution.passed,
                    farewell=checklist.farewell.passed,
                ),
            )

        compliance = None
        if result.compliance is not None:
            compliance = PublicComplianceResult(
                passed=result.compliance.passed,
                issues=result.compliance.assessment.issues,
            )

        return AnalyzeResponse(
            transcript=result.transcript,
            classification=classification,
            quality_score=quality_score,
            compliance=compliance,
            summary=result.summary.summary if result.summary else None,
            action_items=(
                [item.action for item in result.summary.action_items]
                if result.summary else []
            ),
            agent_errors=getattr(result, "agent_errors", {}),
        )
