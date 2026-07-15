import pytest

from app.agents.quality import QualityAgent
from app.schemas import (
    QualityAssessment,
    QualityChecklist,
    QualityCriterionResult,
    TranscriptSegment,
)


class FakeLLMClient:
    def __init__(
        self,
        result: QualityAssessment,
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
    ) -> QualityAssessment:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
                "temperature": temperature,
            }
        )

        return self.result


def make_result() -> QualityAssessment:
    return QualityAssessment(
        checklist=QualityChecklist(
            greeting=QualityCriterionResult(
                passed=True,
                reason="Оператор поздоровался.",
                evidence=None,
            ),
            need_identification=QualityCriterionResult(
                passed=True,
                reason="Оператор уточнил запрос.",
                evidence=None,
            ),
            solution=QualityCriterionResult(
                passed=True,
                reason="Оператор предоставил решение.",
                evidence=None,
            ),
            farewell=QualityCriterionResult(
                passed=False,
                reason="Прощание отсутствует.",
                evidence=None,
            ),
        )
    )


@pytest.mark.asyncio
async def test_quality_agent_calls_llm_with_schema() -> None:
    expected_result = make_result()
    llm_client = FakeLLMClient(
        expected_result
    )

    agent = QualityAgent(llm_client)

    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text="Добрый день, чем могу помочь?",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=3.2,
            end=6.0,
            text="Не работает карта.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=6.2,
            end=9.0,
            text="Уточните, карта заблокирована?",
        ),
    ]

    result = await agent.run(transcript)

    assert result == expected_result
    assert len(llm_client.calls) == 1

    call = llm_client.calls[0]

    assert (
        call["response_model"]
        is QualityAssessment
    )
    assert call["temperature"] == 0.0
    assert "Оператор" in call["user_prompt"]
    assert "Не работает карта" in call["user_prompt"]
    assert "greeting" in call["system_prompt"]
    assert "solution" in call["system_prompt"]


@pytest.mark.asyncio
async def test_quality_agent_preserves_russian_text() -> None:
    llm_client = FakeLLMClient(
        make_result()
    )

    agent = QualityAgent(llm_client)

    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=2.0,
            text="Здравствуйте, хочу узнать про кредит.",
        ),
    ]

    await agent.run(transcript)

    prompt = llm_client.calls[0][
        "user_prompt"
    ]

    assert (
        "Здравствуйте, хочу узнать про кредит."
        in prompt
    )
    assert "\\u0417" not in prompt


@pytest.mark.asyncio
async def test_quality_agent_uses_only_one_llm_call() -> None:
    llm_client = FakeLLMClient(
        make_result()
    )

    agent = QualityAgent(llm_client)

    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=1.0,
            text="Добрый день.",
        ),
    ]

    await agent.run(transcript)

    assert len(llm_client.calls) == 1