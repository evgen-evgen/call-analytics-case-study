import asyncio

from app.agents import (
    AnalysisChatAgent,
    ClassificationAgent,
    ComplianceAgent,
    QualityAgent,
    SummarizerAgent,
)
from app.schemas import (
    ClassificationResult,
    ComplianceResult,
    QualityResult,
    SummaryResult,
    TranscriptSegment,
)


def test_agents_define_prompts_and_result_schemas() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=2.0,
            text="Хочу уточнить условия кредита.",
        )
    ]
    cases = (
        (ClassificationAgent, ClassificationResult),
        (QualityAgent, QualityResult),
        (ComplianceAgent, ComplianceResult),
        (SummarizerAgent, SummaryResult),
    )

    for agent_type, result_model in cases:
        assert agent_type.result_model is result_model
        assert agent_type.system_prompt
        prompt = agent_type.build_user_prompt(
            object.__new__(agent_type),
            transcript,
        )
        assert "условия кредита" in prompt


class FakeTextLLMClient:
    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        assert "анализу банковских звонков" in system_prompt
        assert "Карта была заблокирована" in user_prompt
        assert "Что сделал оператор?" in user_prompt
        assert temperature == 0.1
        return "Оператор заблокировал карту."


def test_chat_agent_answers_from_analysis_context() -> None:
    agent = AnalysisChatAgent(FakeTextLLMClient())

    result = asyncio.run(
        agent.run(
            question="Что сделал оператор?",
            analysis_context="Карта была заблокирована.",
        )
    )

    assert result == "Оператор заблокировал карту."


class FakeGeneralTextLLMClient:
    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        assert "анализ звонка ещё не выполнен" in user_prompt.lower()
        assert "что ты умеешь?" in user_prompt.lower()
        return "Загрузите аудио, и я проанализирую звонок."


def test_chat_agent_answers_without_previous_analysis() -> None:
    agent = AnalysisChatAgent(FakeGeneralTextLLMClient())

    result = asyncio.run(
        agent.run(question="Что ты умеешь?")
    )

    assert "Загрузите аудио" in result
