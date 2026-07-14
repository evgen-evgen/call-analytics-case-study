from app.agents.base import BaseAgent
from app.schemas import SummaryResult, TranscriptSegment


SUMMARIZER_SYSTEM_PROMPT = """
Ты суммаризируешь звонки клиентов банка.

Кратко опиши причину обращения, важные факты и итог.
В action_items включи только явно требующиеся следующие действия.
Не придумывай отсутствующие в транскрипте детали.
Верни результат строго по переданной JSON Schema.
""".strip()


class SummarizerAgent(BaseAgent[SummaryResult]):
    result_model = SummaryResult
    system_prompt = SUMMARIZER_SYSTEM_PROMPT

    def build_user_prompt(
        self,
        transcript: list[TranscriptSegment],
    ) -> str:
        return (
            "Суммаризируй следующий разговор:\n\n"
            f"{self.format_transcript(transcript)}"
        )
