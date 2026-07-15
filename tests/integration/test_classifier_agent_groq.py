"""Integration checks against the configured LLM provider.

Run explicitly so the regular unit-test suite stays deterministic:

    pytest -q -m integration
"""

import asyncio
import os

import pytest

from app.agents.classifier import ClassificationAgent
from app.llm.client import LLMClient
from app.schemas import CallPriority, CallTopic, TranscriptSegment


TEST_CASES = [
    ("Хочу узнать условия кредита.", CallTopic.CREDITS, CallPriority.LOW),
    (
        "Не могу внести платёж по кредиту.",
        CallTopic.CREDITS,
        CallPriority.MEDIUM,
    ),
    (
        "С карты списали деньги без моего согласия.",
        CallTopic.CARDS,
        CallPriority.HIGH,
    ),
    ("Когда будет готова новая карта?", CallTopic.CARDS, CallPriority.LOW),
    (
        "Перевод не дошёл получателю.",
        CallTopic.TRANSFERS,
        CallPriority.MEDIUM,
    ),
    (
        "Как сделать международный перевод?",
        CallTopic.TRANSFERS,
        CallPriority.LOW,
    ),
    (
        "Оформите жалобу на сотрудника.",
        CallTopic.COMPLAINTS,
        CallPriority.MEDIUM,
    ),
]


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("LLM_API_KEY"),
        reason="LLM_API_KEY is not configured",
    ),
]


@pytest.mark.parametrize(
    ("text", "expected_topic", "expected_priority"),
    TEST_CASES,
    ids=[
        "credit-conditions",
        "credit-payment-problem",
        "unauthorized-card-charge",
        "card-issue-status",
        "transfer-delayed",
        "international-transfer-info",
        "employee-complaint",
    ],
)
def test_live_classifier(
    text: str,
    expected_topic: CallTopic,
    expected_priority: CallPriority,
) -> None:
    async def scenario() -> None:
        agent = ClassificationAgent(LLMClient())
        result = await agent.run(
            [
                TranscriptSegment(
                    speaker="Клиент",
                    start=0.0,
                    end=1.0,
                    text=text,
                )
            ]
        )

        assert result.topic == expected_topic, result.reasoning
        assert result.priority == expected_priority, result.reasoning

    asyncio.run(scenario())
