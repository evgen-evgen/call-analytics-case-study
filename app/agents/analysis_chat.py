from app.llm.client import LLMClient


class AnalysisChatAgent:
    SYSTEM_PROMPT = """
Ты помощник по анализу банковских звонков и работе системы.

Если передан анализ звонка, отвечай только на его основании.
Если анализа нет, отвечай на обычные вопросы о возможностях системы
и предлагай загрузить аудио, когда для ответа нужен конкретный звонок.

Правила:
- не придумывай отсутствующие факты;
- при упоминании конкретной реплики указывай спикера и таймкод;
- если информации недостаточно, прямо сообщи об этом;
- отвечай на языке пользователя;
- не изменяй результаты compliance и quality без оснований;
- будь кратким, конкретным и вежливым.
""".strip()

    def __init__(
        self,
        llm_client: LLMClient,
    ) -> None:
        self.llm_client = llm_client

    async def run(
        self,
        *,
        question: str,
        analysis_context: str | None = None,
    ) -> str:
        context = (
            analysis_context
            if analysis_context is not None
            else "Анализ звонка ещё не выполнен."
        )

        return await self.llm_client.generate_text(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                "Результат анализа звонка:\n\n"
                f"{context}\n\n"
                "Вопрос пользователя:\n\n"
                f"{question}"
            ),
            temperature=0.1,
        )
