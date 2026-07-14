from app.agents.base import BaseAgent
from app.schemas import ComplianceResult, TranscriptSegment


COMPLIANCE_SYSTEM_PROMPT = """
Ты проверяешь банковсий разговор на compliance-риски.

Найди недопустимые обещания, запросы секретных данных,
нарушения процедур и другие явные риски. Для каждого нарушения
приведи категорию, описание и, если возможно, цитату с таймкодом.
Если нарушений нет, passed=true. Не придумывай факты.
Верни результат строго по переданной JSON Schema.
""".strip()


class ComplianceAgent(BaseAgent[ComplianceResult]):
    result_model = ComplianceResult
    system_prompt = COMPLIANCE_SYSTEM_PROMPT

    def build_user_prompt(
        self,
        transcript: list[TranscriptSegment],
    ) -> str:
        return (
            "Проверь разговор на compliance-риски:\n\n"
            f"{self.format_transcript(transcript)}"
        )
