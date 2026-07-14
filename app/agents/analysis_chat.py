import json
import os
import re

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
        max_context_chars: int | None = None,
        max_question_chars: int = 8_000,
    ) -> None:
        self.llm_client = llm_client
        self.max_context_chars = max_context_chars or int(
            os.getenv("CHAT_CONTEXT_MAX_CHARS", "40000")
        )
        self.max_question_chars = max_question_chars

    async def run(
        self,
        *,
        question: str,
        analysis_context: str | None = None,
    ) -> str:
        context = self._prepare_analysis_context(
            analysis_context,
            question,
        )
        limited_question = self._limit_text(
            question,
            self.max_question_chars,
            "часть длинного вопроса пропущена",
        )

        return await self.llm_client.generate_text(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                "Результат анализа звонка:\n\n"
                f"{context}\n\n"
                "Вопрос пользователя:\n\n"
                f"{limited_question}"
            ),
            temperature=0.1,
        )

    def _prepare_analysis_context(
        self,
        analysis_context: str | None,
        question: str,
    ) -> str:
        if analysis_context is None:
            return "Анализ звонка ещё не выполнен."

        if len(analysis_context) <= self.max_context_chars:
            return analysis_context

        parsed = self._extract_analysis_json(analysis_context)
        if parsed is not None:
            transcript = parsed.pop("transcript", [])
            analysis_data: object = parsed
        else:
            transcript = self._extract_transcript_json(analysis_context)
            analysis_data = self._remove_transcript_block(analysis_context)

        if transcript is None:
            return self._limit_text(
                analysis_context,
                self.max_context_chars,
                "часть длинного анализа пропущена",
            )
        if not isinstance(transcript, list):
            transcript = []

        selected = self._select_transcript_segments(
            transcript,
            question,
        )
        compact = {
            "analysis": analysis_data,
            "relevant_transcript_segments": selected,
            "transcript_note": (
                "Для длинного звонка выбраны релевантные вопросу "
                "реплики и соседний контекст."
            ),
        }
        serialized = json.dumps(
            compact,
            ensure_ascii=False,
            indent=2,
        )
        return self._limit_text(
            serialized,
            self.max_context_chars,
            "часть релевантных реплик пропущена",
        )

    @staticmethod
    def _extract_analysis_json(value: str) -> dict | None:
        match = re.search(
            r"```json\s*(\{.*\})\s*```",
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            return None

        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _extract_transcript_json(value: str) -> list | None:
        match = re.search(
            r"## Транскрипция\s*```json\s*(\[.*?\])\s*```",
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None

    @staticmethod
    def _remove_transcript_block(value: str) -> str:
        return re.sub(
            r"## Транскрипция\s*```json\s*\[.*?\]\s*```",
            "",
            value,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    @staticmethod
    def _select_transcript_segments(
        transcript: list,
        question: str,
        max_segments: int = 24,
    ) -> list:
        if not transcript:
            return []

        query_terms = {
            term
            for term in re.findall(r"[\w-]+", question.casefold())
            if len(term) >= 3
        }
        scored: list[tuple[int, int]] = []
        for index, segment in enumerate(transcript):
            searchable = json.dumps(
                segment,
                ensure_ascii=False,
            ).casefold()
            score = sum(term in searchable for term in query_terms)
            if score:
                scored.append((score, index))

        if scored:
            selected_indexes: set[int] = set()
            for _, index in sorted(scored, reverse=True):
                selected_indexes.update(
                    range(
                        max(0, index - 1),
                        min(len(transcript), index + 2),
                    )
                )
                if len(selected_indexes) >= max_segments:
                    break
        else:
            edge_size = min(max_segments // 2, len(transcript))
            selected_indexes = set(range(edge_size))
            selected_indexes.update(
                range(max(0, len(transcript) - edge_size), len(transcript))
            )

        return [
            transcript[index]
            for index in sorted(selected_indexes)[:max_segments]
        ]

    @staticmethod
    def _limit_text(
        value: str,
        max_chars: int,
        omission_message: str,
    ) -> str:
        if len(value) <= max_chars:
            return value

        marker = f"\n\n[... {omission_message} ...]\n\n"
        available = max(max_chars - len(marker), 2)
        head_length = available // 2
        tail_length = available - head_length
        return (
            value[:head_length]
            + marker
            + value[-tail_length:]
        )
