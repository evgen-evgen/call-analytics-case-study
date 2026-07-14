from app.agents.base import BaseAgent
from app.schemas import (
    ClassificationResult,
    TranscriptSegment,
)


CLASSIFIER_SYSTEM_PROMPT = """
Ты анализируешь телефонные обращения клиентов банка.

Твоя задача:
1. Определить основную тему обращения.
2. Определить приоритет обработки.

Допустимые темы:
- кредиты
- карты
- переводы
- жалобы
- другое

Правила приоритета:
- high: возможное мошенничество, потеря денег,
  заблокированный доступ, срочная жалоба,
  угроза безопасности или существенный финансовый риск;
- medium: проблема требует действий банка или последующий контакт,
  но нет немедленной угрозы;
- low: информационный вопрос,
  консультация или обычное уточнение.

Используй только информацию из транскрипта.
Не придумывай отсутствующие обстоятельства.
Если обсуждаются несколько тем, выбери основную.
Верни результат строго по переданной JSON Schema.
""".strip()


class ClassificationAgent(
    BaseAgent[ClassificationResult]
):
    result_model = ClassificationResult
    system_prompt = CLASSIFIER_SYSTEM_PROMPT

    def build_user_prompt(
        self,
        transcript: list[TranscriptSegment],
    ) -> str:
        return (
            "Классифицируй следующий разговор:\n\n"
            f"{self.format_transcript(transcript)}"
        )
