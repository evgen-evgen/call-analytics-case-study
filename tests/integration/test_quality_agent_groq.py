import os

import pytest

from app.agents.quality import QualityAgent
from app.llm.client import LLMClient
from app.services.quality_service import (
    calculate_quality_result,
    validate_quality_evidence,
)
from app.schemas import TranscriptSegment


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("LLM_API_KEY"),
        reason="LLM_API_KEY is not configured",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "transcript",
        "expected_score",
    ),
    [
        (
            [
                TranscriptSegment(
                    speaker="Оператор",
                    start=0.0,
                    end=2.0,
                    text=(
                        "Добрый день, "
                        "чем могу помочь?"
                    ),
                ),
                TranscriptSegment(
                    speaker="Клиент",
                    start=2.2,
                    end=5.0,
                    text=(
                        "Хочу узнать ставку "
                        "по кредиту."
                    ),
                ),
                TranscriptSegment(
                    speaker="Оператор",
                    start=5.2,
                    end=7.0,
                    text=(
                        "Вас интересует "
                        "потребительский кредит?"
                    ),
                ),
                TranscriptSegment(
                    speaker="Клиент",
                    start=7.2,
                    end=8.0,
                    text="Да.",
                ),
                TranscriptSegment(
                    speaker="Оператор",
                    start=8.2,
                    end=12.0,
                    text=(
                        "Ставка начинается "
                        "от двенадцати процентов."
                    ),
                ),
                TranscriptSegment(
                    speaker="Оператор",
                    start=12.2,
                    end=14.0,
                    text=(
                        "Спасибо за обращение, "
                        "хорошего дня."
                    ),
                ),
            ],
            100,
        ),
        (
            [
                TranscriptSegment(
                    speaker="Оператор",
                    start=0.0,
                    end=1.0,
                    text="Алло.",
                ),
                TranscriptSegment(
                    speaker="Клиент",
                    start=1.2,
                    end=4.0,
                    text="Не работает карта.",
                ),
                TranscriptSegment(
                    speaker="Оператор",
                    start=4.2,
                    end=6.0,
                    text="Попробуйте позже.",
                ),
                TranscriptSegment(
                    speaker="Оператор",
                    start=6.2,
                    end=7.0,
                    text="Ладно.",
                ),
            ],
            0,
        ),
        (
            [
                TranscriptSegment(
                    speaker="Клиент",
                    start=0.0,
                    end=2.0,
                    text="Перевод не дошёл.",
                ),
                TranscriptSegment(
                    speaker="Оператор",
                    start=2.2,
                    end=4.0,
                    text="Когда вы его отправили?",
                ),
                TranscriptSegment(
                    speaker="Клиент",
                    start=4.2,
                    end=5.0,
                    text="Вчера.",
                ),
                TranscriptSegment(
                    speaker="Оператор",
                    start=5.2,
                    end=9.0,
                    text=(
                        "Я зарегистрировал обращение, "
                        "ответ поступит в течение "
                        "двух рабочих дней."
                    ),
                ),
            ],
            50,
        ),
    ],
)
async def test_real_quality_agent(
    transcript: list[TranscriptSegment],
    expected_score: int,
) -> None:
    client = LLMClient()
    agent = QualityAgent(client)

    assessment = await agent.run(
        transcript
    )

    assessment = validate_quality_evidence(
        assessment,
        transcript,
    )

    result = calculate_quality_result(
        assessment
    )

    assert result.score == expected_score
