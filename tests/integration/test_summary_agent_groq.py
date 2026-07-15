import os

import pytest

from app.agents.summarizer import (
    SummarizerAgent,
)
from app.llm.client import LLMClient
from app.schemas import (
    ActionItemOwner,
    ActionItemStatus,
    TranscriptSegment,
)
from app.services.summary_service import (
    normalize_summary_result,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("LLM_API_KEY"),
        reason="LLM_API_KEY is not configured",
    ),
]


@pytest.mark.asyncio
async def test_creates_pending_bank_action() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=3.0,
            text="Перевод не дошёл получателю.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=3.2,
            end=5.0,
            text="Когда вы его отправили?",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=5.2,
            end=6.0,
            text="Вчера.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=6.2,
            end=11.0,
            text=(
                "Я зарегистрировал обращение. "
                "Ответ поступит в течение "
                "двух рабочих дней."
            ),
        ),
    ]

    client = LLMClient()
    agent = SummarizerAgent(client)

    assessment = await agent.run(
        transcript
    )

    result = normalize_summary_result(
        assessment
    )

    assert 1 <= len(result.action_items) <= 2

    action = result.action_items[0]

    assert action.owner in {
        ActionItemOwner.BANK,
        ActionItemOwner.OPERATOR,
    }
    assert (
        action.status
        == ActionItemStatus.IN_PROGRESS
    )
    assert action.deadline is not None


@pytest.mark.asyncio
async def test_finished_informational_call_has_no_actions() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=2.0,
            text="Добрый день, чем могу помочь?",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=2.2,
            end=5.0,
            text=(
                "Какая ставка по "
                "потребительскому кредиту?"
            ),
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=5.2,
            end=8.0,
            text=(
                "Ставка начинается "
                "от двенадцати процентов."
            ),
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=8.2,
            end=9.0,
            text="Спасибо, это всё.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=9.2,
            end=11.0,
            text="Всего доброго.",
        ),
    ]

    client = LLMClient()
    agent = SummarizerAgent(client)

    assessment = await agent.run(
        transcript
    )

    result = normalize_summary_result(
        assessment
    )

    assert result.action_items == []


@pytest.mark.asyncio
async def test_client_action_for_card_blocking() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=2.0,
            text="Я потерял карту.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=2.2,
            end=6.0,
            text=(
                "Откройте приложение и "
                "заблокируйте карту в разделе Карты."
            ),
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=6.2,
            end=7.0,
            text="Хорошо.",
        ),
    ]

    client = LLMClient()
    agent = SummarizerAgent(client)

    assessment = await agent.run(
        transcript
    )

    result = normalize_summary_result(
        assessment
    )

    assert result.action_items

    assert any(
        item.owner == ActionItemOwner.CLIENT
        and item.status == ActionItemStatus.PENDING
        for item in result.action_items
    )
