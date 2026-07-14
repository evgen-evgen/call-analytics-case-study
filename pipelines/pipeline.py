"""
title: MTBank ASR Debug
author: Evgeni Basov
date: 2026-07-11
version: 0.1.0
license: MIT
description: MTBank ASR pipeline for processing OpenWebUI audio attachments.
"""
import os
import logging
import asyncio
import queue
import threading
from time import perf_counter
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Union

import httpx
from pydantic import BaseModel, Field
from app.schemas import AudioSource

from app.audio_source import (
    AudioSourceError,
    OpenWebUIAudioDownloader,
    find_audio_source,
)

from app.transcriber import (
    Transcriber,
    TranscriptionError,
)

from app.diarizer import (
    Diarizer,
    DiarizationError,
)

from app.aligner import (
    AlignmentError,
    TranscriptAligner,
)

from app.audio_normalizer import (
    AudioNormalizer,
    AudioNormalizationError,
)

from app.observability import (
    bind_request_id,
    configure_logging,
    log_event,
    operation,
    reset_request_id,
)

from app.role_mapper import SpeakerRoleMapper
from app.agents.supervisor import AnalysisResult, AnalysisSupervisor
from app.schemas import TranscriptSegment
from app.llm.client import LLMClient, LLMRequestError
from app.agents.analysis_chat import AnalysisChatAgent
from app.response_presenter import ResponsePresenter



ANALYSIS_MARKER = "<!-- MTBANK_ANALYSIS_RESULT -->"

class Pipeline:
    class Valves(BaseModel):
        WHISPER_MODEL: str = Field(
            default="medium",
            description="Whisper model that will be used later.",
        )

        WHISPER_DEVICE: str = Field(
            default="cpu",
            description="Inference device: cpu or cuda.",
        )

        WHISPER_COMPUTE_TYPE: str = Field(
            default="int8",
            description="Compute type, for example int8 or float16.",
        )

        WHISPER_LANGUAGE: str = Field(
            default="ru",
            description="Audio language. Use an empty value for auto-detection.",
        )

        OPENWEBUI_BASE_URL: str = Field(
            default="http://open-webui:8080",
            description="Internal OpenWebUI URL.",
        )

        DEBUG_OUTPUT_MAX_LENGTH: int = Field(
            default=20_000,
            description="Maximum number of characters returned in debug output.",
        )
        DIARIZATION_MODEL: str = Field(
             default="pyannote/speaker-diarization-community-1",
        )
        DIARIZATION_DEVICE: str = Field(
            default="cpu",
        )

        MAX_CONCURRENT_AUDIO_JOBS: int = Field(
            default=1,
            ge=1,
            description=(
                "Maximum number of audio pipelines processed concurrently."
            ),
        )

    def __init__(self) -> None:
        configure_logging()
        self.id = "mtbank-asr"
        self.name = "MTBank ASR"
        self.valves = self.Valves(
            WHISPER_MODEL=os.getenv("WHISPER_MODEL", "medium"),
            WHISPER_DEVICE=os.getenv("WHISPER_DEVICE", "cpu"),
            WHISPER_COMPUTE_TYPE=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            WHISPER_LANGUAGE=os.getenv("WHISPER_LANGUAGE", "ru"),
            OPENWEBUI_BASE_URL=os.getenv("OPENWEBUI_BASE_URL", "http://open-webui:8080"),
            DEBUG_OUTPUT_MAX_LENGTH=int(os.getenv("DEBUG_OUTPUT_MAX_LENGTH", "20000")),
            DIARIZATION_MODEL=os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"),
            DIARIZATION_DEVICE=os.getenv("DIARIZATION_DEVICE", "cpu"),
            MAX_CONCURRENT_AUDIO_JOBS=int(
                os.getenv("MAX_CONCURRENT_AUDIO_JOBS", "1")
            ),
        )
        self.downloader: OpenWebUIAudioDownloader | None = None
        self.transcriber: Transcriber | None = None
        self.diarizer: Diarizer | None = None
        self.aligner = TranscriptAligner (merge_gap_seconds=1.0)
        self.audio_normalizer = AudioNormalizer(
                        sample_rate=16_000,
                        channels=1,
        )
        self.role_mapper = SpeakerRoleMapper()
        self.llm_client: LLMClient | None = None
        self.supervisor: AnalysisSupervisor | None = None
        self.chat_agent: AnalysisChatAgent | None = None
        self.presenter = ResponsePresenter()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._audio_semaphore: asyncio.Semaphore | None = None

    async def on_startup(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        self._audio_semaphore = asyncio.Semaphore(
            self.valves.MAX_CONCURRENT_AUDIO_JOBS
        )
        self.llm_client = LLMClient()
        self.supervisor = AnalysisSupervisor(self.llm_client)
        self.chat_agent = AnalysisChatAgent(self.llm_client)

        self.downloader = OpenWebUIAudioDownloader(
            base_url=self.valves.OPENWEBUI_BASE_URL,
        )

        self.transcriber = Transcriber(
            model_name=self.valves.WHISPER_MODEL,
            device=self.valves.WHISPER_DEVICE,
            compute_type=self.valves.WHISPER_COMPUTE_TYPE,
            language=self.valves.WHISPER_LANGUAGE,
        )

        self.diarizer = Diarizer(
            model_name=self.valves.DIARIZATION_MODEL,
            device=self.valves.DIARIZATION_DEVICE,
            hf_token=os.getenv("HF_TOKEN"),
            num_speakers=2,
        )
        log_event(
            "pipeline.starting",
            whisper_model=self.valves.WHISPER_MODEL,
            whisper_device=self.valves.WHISPER_DEVICE,
            diarization_model=self.valves.DIARIZATION_MODEL,
            diarization_device=self.valves.DIARIZATION_DEVICE,
            max_concurrent_audio_jobs=(
                self.valves.MAX_CONCURRENT_AUDIO_JOBS
            ),
        )

        with operation(
            "model.load.whisper",
            model=self.valves.WHISPER_MODEL,
            device=self.valves.WHISPER_DEVICE,
        ):
            self.transcriber.load()

        with operation(
            "model.load.diarization",
            model=self.valves.DIARIZATION_MODEL,
            device=self.valves.DIARIZATION_DEVICE,
        ):
            self.diarizer.load()

        log_event("pipeline.started")

    async def on_shutdown(self) -> None:
        if self.transcriber is not None:
            self.transcriber.unload()
        if self.diarizer is not None:
            self.diarizer.unload()
        if self.llm_client is not None:
            await self.llm_client.client.close()
        self._audio_semaphore = None
        self._event_loop = None

        log_event("pipeline.stopped")


    async def on_valves_updated(self) -> None:
        """
        Recreate inference components after configuration changes.
        """
        self.downloader = OpenWebUIAudioDownloader(
            base_url=self.valves.OPENWEBUI_BASE_URL,
        )

        language = (
            self.valves.WHISPER_LANGUAGE.strip()
            or None
        )

        self.transcriber = Transcriber(
            model_name=self.valves.WHISPER_MODEL,
            device=self.valves.WHISPER_DEVICE,
            compute_type=self.valves.WHISPER_COMPUTE_TYPE,
            language=language,
        )

        self.transcriber.load()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> Union[str, Generator, Iterator]:
        if self.downloader is None:
            return "Pipeline не был корректно инициализирован."

        if self.transcriber is None:
            return "Ошибка faster-whisper не инициализирован."

        if self.diarizer is None:
            return "Ошибка diarizer не инициализирован."

        try:
            audio_source = find_audio_source(user_message)
        except AudioSourceError as exc:
            return self._format_error(exc)

        if audio_source is not None and not body.get("stream"):
            log_event(
                "request.audio_skipped",
                reason="non_stream_request",
                filename=audio_source.filename,
            )
            return (
                "Аудиофайл будет обработан "
                "в основном streaming-запросе."
            )

        internal_task = self._internal_task(body)
        if internal_task is not None:
            return self._run_async(
                self._answer_internal_task(
                    user_message=user_message,
                    task=internal_task,
                )
            )

        if body.get("stream"):
            return self._stream_async(
                user_message=user_message,
                model_id=model_id,
                messages=messages,
                body=body,
            )

        return self._run_async(
            self._pipe_async(
                user_message=user_message,
                model_id=model_id,
                messages=messages,
                body=body,
            )
        )

    async def _pipe_async(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:

        request_token = bind_request_id(str(uuid4()))
        request_started_at = perf_counter()
        outcome = "failed"

        log_event(
            "request.started",
            model_id=model_id,
            stream=bool(body.get("stream")),
        )

        try:
            with operation("audio.source.extract"):
                source = find_audio_source(user_message)

            if source is not None:
                result = await self._process_audio_with_limit(
                    source,
                    progress=progress,
                    cancel_event=cancel_event,
                )
            else:
                previous_analysis = self._extract_latest_analysis(
                    messages
                )

                if previous_analysis is not None:
                    result = await self._answer_text(
                        user_message=user_message,
                        analysis=previous_analysis,
                    )
                elif user_message.strip():
                    result = await self._answer_text(
                        user_message=user_message,
                    )
                else:
                    result = self._format_welcome_message()

            outcome = "completed"
            return result

        except asyncio.CancelledError:
            outcome = "cancelled"
            log_event("request.cancelled")
            raise

        except Exception as exc:
            return self._format_error(exc)

        finally:
            log_event(
                "request.finished",
                outcome=outcome,
                duration_ms=round(
                    (perf_counter() - request_started_at) * 1000,
                    2,
                ),
            )
            reset_request_id(request_token)

    async def _process_audio_with_limit(
        self,
        source: AudioSource,
        *,
        progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        if self._audio_semaphore is None:
            self._audio_semaphore = asyncio.Semaphore(
                self.valves.MAX_CONCURRENT_AUDIO_JOBS
            )

        semaphore = self._audio_semaphore

        if semaphore.locked():
            self._report_progress(progress, "audio_queued")
            log_event("audio.queue.waiting")

        with operation("audio.queue.wait"):
            await semaphore.acquire()

        log_event("audio.queue.acquired")
        try:
            self._raise_if_cancelled(cancel_event)
            return await self._process_audio(
                source,
                progress=progress,
                cancel_event=cancel_event,
            )
        finally:
            semaphore.release()
            log_event("audio.queue.released")

    @staticmethod
    def _format_error(exc: Exception) -> str:
        if isinstance(exc, AudioSourceError):
            return (
                "## Ошибка входного файла\n\n"
                f"{exc}"
            )

        if isinstance(exc, TranscriptionError):
            return (
                "## Ошибка транскрибации\n\n"
                f"{exc}"
            )

        if isinstance(exc, httpx.HTTPStatusError):
            return (
                "## Ошибка загрузки файла\n\n"
                f"OpenWebUI API status: "
                f"`{exc.response.status_code}`\n\n"
                f"Response: `{exc.response.text[:500]}`"
            )

        if isinstance(exc, httpx.RequestError):
            return (
                "## Ошибка подключения к OpenWebUI\n\n"
                f"`{exc}`"
            )

        if isinstance(exc, DiarizationError):
            return (
                "## Ошибка диаризации\n\n"
                f"{exc}"
            )

        if isinstance(exc, AlignmentError):
            return (
                "## Ошибка выравнивания транскрипта\n\n"
                f"{exc}"
            )

        if isinstance(exc, AudioNormalizationError):
            return (
                "## Ошибка подготовки аудио\n\n"
                f"{exc}"
            )

        if isinstance(exc, LLMRequestError):
            log_event(
                "request.llm_error",
                level=logging.ERROR,
                error_message=str(exc),
            )
            if "reduce the length" in str(exc).lower():
                return (
                    "## Слишком длинный контекст\n\n"
                    "История разговора превысила лимит модели. "
                    "Попробуйте задать более короткий вопрос или "
                    "начать новый чат."
                )
            return (
                "## Ошибка модели\n\n"
                "Модель не смогла обработать запрос. "
                "Попробуйте повторить его ещё раз."
            )

        log_event(
            "request.unexpected_error",
            level=logging.ERROR,
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return (
            "## Неожиданная ошибка\n\n"
            f"Type: `{type(exc).__name__}`\n\n"
            f"Message: `{exc}`"
        )

    async def _process_audio(
        self,
        source: AudioSource,
        *,
        progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        downloaded_path: Path | None = None
        normalized_path: Path | None = None

        try:
            self._raise_if_cancelled(cancel_event)
            self._report_progress(progress, "file_received")

            with operation(
                "audio.download",
                content_type=source.content_type,
            ):
                downloaded_path = await asyncio.to_thread(
                    self.downloader.download,
                    source,
                )

            with operation("audio.normalize"):
                normalized_path = await asyncio.to_thread(
                    self.audio_normalizer.normalize,
                    downloaded_path,
                )

            self._report_progress(progress, "audio_prepared")
            self._raise_if_cancelled(cancel_event)

            with operation(
                "audio.transcribe",
                model=self.valves.WHISPER_MODEL,
            ):
                raw_transcript = await asyncio.to_thread(
                    self.transcriber.run,
                    normalized_path,
                    cancel_event,
                )

            self._report_progress(
                progress,
                "transcription_completed",
            )
            self._raise_if_cancelled(cancel_event)

            log_event(
                "transcript.created",
                segment_count=len(raw_transcript),
                word_count=sum(
                    len(segment.words)
                    for segment in raw_transcript
                ),
            )

            with operation(
                "audio.diarize",
                model=self.valves.DIARIZATION_MODEL,
            ):
                diarization = await asyncio.to_thread(
                    self.diarizer.run,
                    normalized_path,
                )

            self._report_progress(
                progress,
                "diarization_completed",
            )
            self._raise_if_cancelled(cancel_event)

            log_event(
                "diarization.created",
                segment_count=len(diarization),
                speaker_count=len({
                    segment.speaker
                    for segment in diarization
                }),
            )

            with operation("transcript.align"):
                aligned_transcript = await asyncio.to_thread(
                    self.aligner.align,
                    transcript=raw_transcript,
                    diarization=diarization,
                )

            with operation("transcript.map_roles"):
                final_transcript = await asyncio.to_thread(
                    self.role_mapper.map_roles,
                    aligned_transcript,
                )

            self._report_progress(progress, "analysis_started")
            self._raise_if_cancelled(cancel_event)
            analysis = await self._run_analysis(
                final_transcript
            )
            self._report_progress(progress, "analysis_completed")

            # response = {
            #     "transcript": [
            #         segment.model_dump()
            #         for segment in final_transcript
            #     ],
            #     "classification": (
            #         analysis.classification.model_dump()
            #     ),
            #     "quality_score": analysis.quality.model_dump(),
            #     "compliance": analysis.compliance.model_dump(),
            #     "summary": analysis.summary.summary,
            #     "action_items": analysis.summary.action_items,
            # }
            return self.presenter.format_analysis(
                analysis,
                ANALYSIS_MARKER,
            )

        finally:
            if downloaded_path is not None:
                downloaded_path.unlink(missing_ok=True)

            if (
                normalized_path is not None
                and normalized_path != downloaded_path
            ):
                normalized_path.unlink(missing_ok=True)

    async def _run_analysis(
        self,
        transcript: list[TranscriptSegment],
    ) -> AnalysisResult:
        if self.supervisor is None:
            raise RuntimeError(
                "Analysis supervisor is not initialized."
            )

        with operation("agents.analyze"):
            return await self.supervisor.run(transcript)

    def _format_welcome_message(self) -> str:
        return (
            "## Анализ банковского звонка\n\n"
            "Загрузите аудиофайл в формате WAV, MP3 или OGG.\n\n"
            "Система выполнит:\n\n"
            "- транскрибацию с временными метками;\n"
            "- разделение на Оператора и Клиента;\n"
            "- классификацию обращения;\n"
            "- оценку качества оператора;\n"
            "- compliance-проверку;\n"
            "- суммаризацию и формирование action items.\n\n"
            "После получения анализа можно задавать "
            "дополнительные вопросы по звонку."
        )

    def _extract_latest_analysis(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue

            content = self._message_content(message)

            if ANALYSIS_MARKER in content:
                return content.replace(
                    ANALYSIS_MARKER,
                    "",
                    1,
                ).strip()

        return None

    @staticmethod
    def _message_content(
        message: dict[str, Any],
    ) -> str:
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "text":
                    parts.append(
                        str(item.get("text", ""))
                    )

            return "\n".join(parts)

        return str(content)

    async def _answer_text(
        self,
        *,
        user_message: str,
        analysis: str | None = None,
    ) -> str:
        if self.chat_agent is None:
            raise RuntimeError(
                "Analysis chat agent is not initialized."
            )

        return await self.chat_agent.run(
            question=user_message,
            analysis_context=analysis,
        )

    async def _answer_internal_task(
        self,
        *,
        user_message: str,
        task: str,
    ) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client is not initialized.")

        log_event(
            "request.internal_task",
            task=task,
        )
        return await self.llm_client.generate_text(
            system_prompt=(
                "Выполни внутреннюю служебную задачу "
                "OpenWebUI строго по инструкции пользователя."
            ),
            user_prompt=user_message,
            temperature=0.0,
        )

    @staticmethod
    def _internal_task(body: dict[str, Any]) -> str | None:
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            return None

        task = metadata.get("task")
        return str(task) if task else None

    def _run_async(self, coroutine: Any) -> Any:
        if self._event_loop is None:
            coroutine.close()
            raise RuntimeError(
                "Pipeline event loop is not initialized."
            )

        return asyncio.run_coroutine_threadsafe(
            coroutine,
            self._event_loop,
        ).result()

    def _stream_async(
        self,
        *,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> Iterator[str]:
        if self._event_loop is None:
            yield "## Ошибка\n\nPipeline event loop is not initialized."
            return

        progress_queue: queue.Queue[str] = queue.Queue()
        cancel_event = threading.Event()
        future = asyncio.run_coroutine_threadsafe(
            self._pipe_async(
                user_message=user_message,
                model_id=model_id,
                messages=messages,
                body=body,
                progress=progress_queue.put,
                cancel_event=cancel_event,
            ),
            self._event_loop,
        )

        try:
            while not future.done() or not progress_queue.empty():
                try:
                    yield progress_queue.get(timeout=0.25)
                except queue.Empty:
                    # Give StreamingResponse a regular opportunity to observe
                    # a disconnected client. Without this heartbeat, one
                    # next() call blocks until transcription produces output,
                    # so Open WebUI's Stop cannot close this generator.
                    yield ""

            yield future.result()
        finally:
            cancel_event.set()
            if not future.done():
                future.cancel()

    def _report_progress(
        self,
        callback: Callable[[str], None] | None,
        event: str,
    ) -> None:
        if callback is not None:
            callback(self.presenter.format_progress(event))

    @staticmethod
    def _raise_if_cancelled(
        cancel_event: threading.Event | None,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
