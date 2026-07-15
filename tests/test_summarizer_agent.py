import pytest

from app.agents.summarizer import SummarizerAgent
from app.schemas import (
    ActionItem,
    ActionItemOwner,
    ActionItemStatus,
    SummaryAssessment,
    TranscriptSegment,
)


class FakeLLMClient:
    def __init__(
        self,
        result: SummaryAssessment,
    ) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model,
        temperature: float,
        response_format_mode: str,
    ) -> SummaryAssessment:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
                "temperature": temperature,
                "response_format_mode": response_format_mode,
            }
        )

        return self.result


def make_result() -> SummaryAssessment:
    return SummaryAssessment(
        summary=(
            "Клиент сообщил о задержанном переводе. "
            "Оператор уточнил дату операции. "
            "Обращение было зарегистрировано. "
            "Результат проверки пока ожидается."
        ),
        action_items=[
            ActionItem(
                action=(
                    "Предоставить клиенту "
                    "результат проверки перевода."
                ),
                owner=ActionItemOwner.BANK,
                status=ActionItemStatus.IN_PROGRESS,
                deadline="В течение двух рабочих дней",
                reason=(
                    "Обращение зарегистрировано, "
                    "но результат ещё не получен."
                ),
            )
        ],
    )


@pytest.mark.asyncio
async def test_summarizer_calls_llm_with_schema() -> None:
    expected_result = make_result()
    llm_client = FakeLLMClient(
        expected_result
    )

    agent = SummarizerAgent(
        llm_client
    )

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
            text="Когда вы отправили перевод?",
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
            end=10.0,
            text=(
                "Я зарегистрировал обращение. "
                "Ответ поступит в течение "
                "двух рабочих дней."
            ),
        ),
    ]

    result = await agent.run(
        transcript
    )

    assert result == expected_result
    assert len(llm_client.calls) == 1

    call = llm_client.calls[0]

    assert (
        call["response_model"]
        is SummaryAssessment
    )
    assert call["temperature"] == 0.0
    assert call["response_format_mode"] == "json_object"
    assert "3–5" in call["system_prompt"]
    assert "action item" in call["system_prompt"].lower()
    assert (
        "Перевод не дошёл получателю"
        in call["user_prompt"]
    )


@pytest.mark.asyncio
async def test_summarizer_preserves_russian_text() -> None:
    llm_client = FakeLLMClient(
        make_result()
    )

    agent = SummarizerAgent(
        llm_client
    )

    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=3.0,
            text=(
                "Здравствуйте, хочу узнать "
                "про потребительский кредит."
            ),
        ),
    ]

    await agent.run(
        transcript
    )

    prompt = llm_client.calls[0][
        "user_prompt"
    ]

    assert (
        "потребительский кредит"
        in prompt
    )
    assert "\\u0417" not in prompt


@pytest.mark.asyncio
async def test_summarizer_uses_one_llm_call() -> None:
    llm_client = FakeLLMClient(
        make_result()
    )

    agent = SummarizerAgent(
        llm_client
    )

    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=1.0,
            text="Добрый день.",
        ),
    ]

    await agent.run(
        transcript
    )

    assert len(llm_client.calls) == 1
