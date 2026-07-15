import asyncio

from app.agents.classifier import ClassificationAgent
from app.schemas import (
    CallPriority,
    CallTopic,
    ClassificationResult,
    TranscriptSegment,
)


class FakeLLMClient:
    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ClassificationResult],
        temperature: float,
    ) -> ClassificationResult:
        assert "пропали деньги с карты" in user_prompt.lower()
        assert "эту операцию не совершал" in user_prompt.lower()
        assert "банковского контакт-центра" in system_prompt.lower()
        assert response_model is ClassificationResult
        assert temperature == 0.0

        return ClassificationResult(
            topic=CallTopic.CARDS,
            priority=CallPriority.HIGH,
            reasoning=(
                "Клиент сообщил о неизвестной операции "
                "и потере денег с карты."
            ),
        )


def test_classifies_suspicious_card_transaction_as_high_priority() -> None:
    agent = ClassificationAgent(
        llm_client=FakeLLMClient(),
    )

    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.4,
            text="Добрый день, банк, меня зовут Анна.",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=3.8,
            end=8.1,
            text=(
                "У меня пропали деньги с карты, "
                "я эту операцию не совершал."
            ),
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=8.5,
            end=12.0,
            text=(
                "Сейчас проверим информацию "
                "и заблокируем карту."
            ),
        ),
    ]

    result = asyncio.run(agent.run(transcript))

    assert result.topic == CallTopic.CARDS
    assert result.priority == CallPriority.HIGH
