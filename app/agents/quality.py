from app.agents.base import BaseAgent
from app.schemas import QualityResult, TranscriptSegment


QUALITY_SYSTEM_PROMPT = """
Ты оцениваешь качество работы оператора банка.

Проверь приветствие, выявление потребности,
предложение решения и прощание. Выставь total от 0 до 100
и кратко перечисли проблемы. Опирайся только на транскрипт.
Верни результат строго по переданной JSON Schema.
""".strip()


class QualityAgent(BaseAgent[QualityResult]):
    result_model = QualityResult
    system_prompt = QUALITY_SYSTEM_PROMPT

    def build_user_prompt(
        self,
        transcript: list[TranscriptSegment],
    ) -> str:
        return (
            "Оцени качество работы оператора:\n\n"
            f"{self.format_transcript(transcript)}"
        )
