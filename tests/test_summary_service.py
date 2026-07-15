import pytest

from app.schemas import (
    ActionItem,
    ActionItemOwner,
    ActionItemStatus,
    SummaryAssessment,
)
from app.services.summary_service import (
    normalize_summary_result,
)


def make_summary(
    *,
    action_items: list[ActionItem] | None = None,
) -> SummaryAssessment:
    return SummaryAssessment(
        summary=(
            "Клиент обратился из-за задержанного перевода. "
            "Оператор уточнил детали операции. "
            "Обращение было зарегистрировано. "
            "Результат проверки пока не получен."
        ),
        action_items=action_items or [],
    )


def test_summary_allows_more_than_five_sentences() -> None:
    summary = " ".join(
        f"Предложение номер {number}."
        for number in range(1, 8)
    )

    result = SummaryAssessment(summary=summary, action_items=[])

    assert result.summary == summary


@pytest.mark.parametrize(
    "summary",
    [
        (
            "Клиент обратился по вопросу кредита. "
            "Оператор уточнил необходимую сумму. "
            "Клиенту были разъяснены условия."
        ),
        (
            "Клиент сообщил о задержке перевода. "
            "Оператор проверил данные операции. "
            "Обращение зарегистрировано. "
            "Результат пока ожидается."
        ),
        (
            "Клиент сообщил о потере карты. "
            "Оператор уточнил обстоятельства. "
            "Клиенту рекомендовали заблокировать карту. "
            "Блокировка во время звонка не выполнялась. "
            "Клиент должен выполнить её самостоятельно."
        ),
    ],
)
def test_summary_accepts_three_to_five_sentences(
    summary: str,
) -> None:
    result = SummaryAssessment(
        summary=summary,
        action_items=[],
    )

    assert result.summary == summary


def test_normalize_summary_removes_duplicate_actions() -> None:
    first = ActionItem(
        action="Предоставить клиенту результат проверки.",
        owner=ActionItemOwner.BANK,
        status=ActionItemStatus.IN_PROGRESS,
        deadline="В течение двух рабочих дней",
        reason="Обращение зарегистрировано.",
    )

    duplicate = ActionItem(
        action="  предоставить клиенту результат проверки. ",
        owner=ActionItemOwner.BANK,
        status=ActionItemStatus.IN_PROGRESS,
        deadline="В течение двух рабочих дней",
        reason="Клиент ожидает ответ.",
    )

    assessment = make_summary(
        action_items=[
            first,
            duplicate,
        ]
    )

    normalized = normalize_summary_result(
        assessment
    )

    assert len(normalized.action_items) == 1
    assert (
        normalized.action_items[0].action
        == first.action
    )


def test_normalize_summary_keeps_different_actions() -> None:
    assessment = make_summary(
        action_items=[
            ActionItem(
                action="Проверить статус перевода.",
                owner=ActionItemOwner.BANK,
                status=ActionItemStatus.IN_PROGRESS,
                deadline=None,
                reason="Перевод не поступил.",
            ),
            ActionItem(
                action="Сообщить клиенту результат проверки.",
                owner=ActionItemOwner.OPERATOR,
                status=ActionItemStatus.PENDING,
                deadline=None,
                reason="Клиент ожидает обратную связь.",
            ),
        ]
    )

    normalized = normalize_summary_result(
        assessment
    )

    assert len(normalized.action_items) == 2


def test_summary_allows_empty_action_items() -> None:
    assessment = make_summary()

    assert assessment.action_items == []
