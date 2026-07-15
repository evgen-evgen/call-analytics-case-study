import asyncio
import json

from app.agents import (
    AnalysisChatAgent,
    ClassificationAgent,
    ComplianceAgent,
    QualityAgent,
    SummarizerAgent,
)
from app.schemas import (
    ClassificationResult,
    ComplianceAssessment,
    QualityAssessment,
    QualityResult,
    SummaryAssessment,
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
        (QualityAgent, QualityAssessment),
        (ComplianceAgent, ComplianceAssessment),
        (SummarizerAgent, SummaryAssessment),
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


class CapturingTextLLMClient:
    def __init__(self) -> None:
        self.user_prompt = ""

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        self.user_prompt = user_prompt
        return "Ответ"


def test_chat_agent_limits_long_analysis_context() -> None:
    client = CapturingTextLLMClient()
    agent = AnalysisChatAgent(
        client,
        max_context_chars=1000,
    )
    context = "НАЧАЛО" + ("x" * 2000) + "КОНЕЦ"

    result = asyncio.run(
        agent.run(
            question="Что произошло?",
            analysis_context=context,
        )
    )

    assert result == "Ответ"
    assert "НАЧАЛО" in client.user_prompt
    assert "КОНЕЦ" in client.user_prompt
    assert "часть длинного анализа пропущена" in client.user_prompt
    assert len(client.user_prompt) < 1400


def test_chat_agent_keeps_analysis_and_relevant_file_segments() -> None:
    client = CapturingTextLLMClient()
    agent = AnalysisChatAgent(client, max_context_chars=1400)
    transcript = [
        {
            "speaker": "Клиент",
            "start": float(index),
            "end": float(index + 1),
            "text": (
                "Кодовое слово альбатрос и спорная операция."
                if index == 50
                else f"Обычная реплика номер {index}."
            ),
        }
        for index in range(100)
    ]
    analysis = {
        "transcript": transcript,
        "classification": {
            "topic": "карты",
            "priority": "high",
            "reasoning": "Неизвестная операция.",
        },
        "quality": {"total": 80},
        "compliance": {"passed": True, "issues": []},
        "summary": {
            "summary": "Клиент оспаривает операцию.",
            "action_items": ["Проверить операцию."],
        },
    }
    context = (
        "## Анализ звонка\n\n```json\n"
        + json.dumps(analysis, ensure_ascii=False)
        + "\n```"
    )

    asyncio.run(
        agent.run(
            question="Что сказано про альбатрос?",
            analysis_context=context,
        )
    )

    assert "Клиент оспаривает операцию" in client.user_prompt
    assert "альбатрос" in client.user_prompt
    assert '"start": 49.0' in client.user_prompt
    assert '"start": 51.0' in client.user_prompt
    assert "Обычная реплика номер 20" not in client.user_prompt


def test_chat_agent_reads_new_transcript_json_format() -> None:
    client = CapturingTextLLMClient()
    agent = AnalysisChatAgent(client, max_context_chars=1000)
    transcript = [
        {
            "speaker": "Оператор",
            "start": float(index),
            "end": float(index + 1),
            "text": (
                "Карта была заблокирована."
                if index == 40
                else f"Реплика {index}."
            ),
        }
        for index in range(80)
    ]
    context = (
        "## Транскрипция\n\n```json\n"
        + json.dumps(transcript, ensure_ascii=False)
        + "\n```\n\n"
        "## Резюме\n\nКлиент сообщил о проблеме с картой."
    )

    asyncio.run(
        agent.run(
            question="Когда карта была заблокирована?",
            analysis_context=context,
        )
    )

    assert "Клиент сообщил о проблеме" in client.user_prompt
    assert "Карта была заблокирована" in client.user_prompt
    assert '"start": 39.0' in client.user_prompt
    assert '"start": 41.0' in client.user_prompt
