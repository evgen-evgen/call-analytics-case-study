import os

import pytest

from app.agents.supervisor import AnalysisSupervisor
from app.llm.client import LLMClient
from app.schemas import TranscriptSegment


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("LLM_API_KEY"),
        reason="LLM_API_KEY is not configured",
    ),
]


@pytest.mark.asyncio
async def test_supervisor_with_real_llm() -> None:
    supervisor = AnalysisSupervisor(
        LLMClient()
    )

    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=2.4,
            text="Добрый день, чем могу помочь?",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=2.6,
            end=5.6,
            text="Перевод не дошёл получателю.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=5.8,
            end=8.0,
            text="Когда вы отправили перевод?",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=8.2,
            end=9.0,
            text="Вчера.",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=9.2,
            end=13.8,
            text=(
                "Я зарегистрировал обращение. "
                "Ответ поступит в течение "
                "двух рабочих дней."
            ),
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=14.0,
            end=16.0,
            text=(
                "Спасибо за обращение, "
                "всего доброго."
            ),
        ),
    ]

    result = await supervisor.run(transcript)

    assert result.classification.topic == "переводы"

    assert result.classification.priority in {
        "low",
        "medium",
        "high",
    }

    assert result.quality.score >= 75
    assert result.compliance is not None
    assert result.summary.summary