import logging
from typing import Any

import httpx

from app.asr import (
    AlignmentError,
    AudioNormalizationError,
    AudioSourceError,
    DiarizationError,
    TranscriptionError,
)
from app.llm.client import LLMClientError
from app.mappers import AnalyzeResponseMapper
from app.observability import log_event
from app.response_presenter import ResponsePresenter
from app.schemas import AnalyzeResponse
from app.services import AudioAnalysisError


ANALYSIS_MARKER = "<!-- MTBANK_ANALYSIS_RESULT -->"


class OpenWebUIResponseFormatter:
    def __init__(self) -> None:
        self.presenter = ResponsePresenter()
        self.response_mapper = AnalyzeResponseMapper()

    def analysis(self, result: Any) -> str:
        response = self.response_mapper.map(result)
        return response.model_dump_json(indent=2)

    def progress(self, event: str) -> str:
        return self.presenter.format_progress(event)

    def latest_analysis(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = self._message_content(message)
            if ANALYSIS_MARKER in content:
                return content.replace(ANALYSIS_MARKER, "", 1).strip()
            try:
                analysis = AnalyzeResponse.model_validate_json(content)
            except ValueError:
                continue
            return analysis.model_dump_json(indent=2)
        return None

    @staticmethod
    def welcome() -> str:
        return (
            "## Анализ банковского звонка\n\n"
            "Загрузите аудиофайл в формате WAV, MP3 или OGG.\n\n"
            "Система выполнит транскрибацию, разделение ролей, "
            "классификацию, оценку качества, compliance-проверку "
            "и суммаризацию."
        )

    @staticmethod
    def audio_skipped() -> str:
        return "Аудиофайл будет обработан в основном streaming-запросе."

    @staticmethod
    def error(exc: Exception) -> str:
        headings = (
            (AudioSourceError, "Ошибка входного файла"),
            (TranscriptionError, "Ошибка транскрибации"),
            (DiarizationError, "Ошибка диаризации"),
            (AlignmentError, "Ошибка выравнивания транскрипта"),
            (AudioNormalizationError, "Ошибка подготовки аудио"),
            (AudioAnalysisError, "Ошибка анализа аудио"),
        )
        for error_type, heading in headings:
            if isinstance(exc, error_type):
                return f"## {heading}\n\n{exc}"

        if isinstance(exc, httpx.HTTPStatusError):
            return (
                "## Ошибка загрузки файла\n\n"
                f"OpenWebUI API status: `{exc.response.status_code}`\n\n"
                f"Response: `{exc.response.text[:500]}`"
            )
        if isinstance(exc, httpx.RequestError):
            return f"## Ошибка подключения к OpenWebUI\n\n`{exc}`"
        if isinstance(exc, LLMClientError):
            log_event(
                "request.llm_error",
                level=logging.ERROR,
                error_type=type(exc).__name__,
                error_message="LLM request failed",
            )
            if "reduce the length" in str(exc).lower():
                return "## Слишком длинный контекст\n\nНачните новый чат."
            return "## Ошибка модели\n\nПопробуйте повторить запрос."

        log_event(
            "request.unexpected_error",
            level=logging.ERROR,
            exc_info=True,
            error_type=type(exc).__name__,
            error_message="Unexpected request processing failure",
        )
        return (
            "## Неожиданная ошибка\n\n"
            f"Type: `{type(exc).__name__}`\n\nMessage: `{exc}`"
        )

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content)
