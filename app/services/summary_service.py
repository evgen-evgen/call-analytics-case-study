from app.schemas import (
    ActionItem,
    SummaryAssessment,
)


def normalize_summary_result(
    assessment: SummaryAssessment,
) -> SummaryAssessment:
    unique_items: list[ActionItem] = []
    seen_actions: set[str] = set()

    for item in assessment.action_items:
        normalized_action = _normalize_text(
            item.action
        )

        if normalized_action in seen_actions:
            continue

        seen_actions.add(normalized_action)
        unique_items.append(item)

    return SummaryAssessment(
        summary=assessment.summary.strip(),
        action_items=unique_items,
    )


def _normalize_text(value: str) -> str:
    return " ".join(
        value.lower().strip().split()
    )