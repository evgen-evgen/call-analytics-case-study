import asyncio
import threading
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from app.agents.analysis_chat import AnalysisChatAgent
from app.asr import OpenWebUIAudioDownloader
from app.llm.client import LLMClient
from app.observability import (
    bind_request_id,
    log_event,
    operation,
    reset_request_id,
)
from app.openwebui.formatter import OpenWebUIResponseFormatter
from app.openwebui.request import OpenWebUIRequest
from app.services import AudioAnalysisService


class OpenWebUIRequestHandler:
    def __init__(
        self,
        *,
        downloader: OpenWebUIAudioDownloader,
        analysis_service: AudioAnalysisService,
        chat_agent: AnalysisChatAgent,
        llm_client: LLMClient,
        formatter: OpenWebUIResponseFormatter,
        max_concurrent_audio_jobs: int,
    ) -> None:
        self.downloader = downloader
        self.analysis_service = analysis_service
        self.chat_agent = chat_agent
        self.llm_client = llm_client
        self.formatter = formatter
        self.audio_semaphore = asyncio.Semaphore(max_concurrent_audio_jobs)

    async def handle(
        self,
        request: OpenWebUIRequest,
        *,
        progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        token = bind_request_id(str(uuid4()))
        started_at = perf_counter()
        outcome = "failed"
        log_event(
            "request.started",
            model_id=request.model_id,
            stream=request.stream,
        )
        try:
            result = await self._dispatch(request, progress, cancel_event)
            outcome = "completed"
            return result
        except asyncio.CancelledError:
            outcome = "cancelled"
            log_event("request.cancelled")
            raise
        except Exception as exc:
            return self.formatter.error(exc)
        finally:
            log_event(
                "request.finished",
                outcome=outcome,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )
            reset_request_id(token)

    async def _dispatch(
        self,
        request: OpenWebUIRequest,
        progress: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
    ) -> str:
        if request.internal_task is not None:
            return await self._answer_internal_task(request)
        if request.audio_source is not None:
            if not request.stream:
                log_event(
                    "request.audio_skipped",
                    reason="non_stream_request",
                    filename=request.audio_source.filename,
                )
                return self.formatter.audio_skipped()
            return await self._handle_audio(request, progress, cancel_event)

        previous = self.formatter.latest_analysis(request.messages)
        if previous is not None or request.user_message.strip():
            return await self.chat_agent.run(
                question=request.user_message,
                analysis_context=previous,
            )
        return self.formatter.welcome()

    async def _handle_audio(
        self,
        request: OpenWebUIRequest,
        progress: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
    ) -> str:
        if self.audio_semaphore.locked():
            self._progress(progress, "audio_queued")
            log_event("audio.queue.waiting")

        with operation("audio.queue.wait"):
            await self.audio_semaphore.acquire()
        try:
            self._raise_if_cancelled(cancel_event)
            return await self._analyze_audio(request, progress, cancel_event)
        finally:
            self.audio_semaphore.release()
            log_event("audio.queue.released")

    async def _analyze_audio(
        self,
        request: OpenWebUIRequest,
        progress: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
    ) -> str:
        source = request.audio_source
        if source is None:
            raise RuntimeError("Audio source is missing.")
        downloaded_path: Path | None = None
        try:
            self._progress(progress, "file_received")
            with operation("audio.download", content_type=source.content_type):
                downloaded_path = await asyncio.to_thread(
                    self.downloader.download,
                    source,
                )
            analysis = await self.analysis_service.analyze_path(
                downloaded_path,
                progress=lambda event: self._progress(progress, event),
                cancel_event=cancel_event,
            )
            return self.formatter.analysis(analysis)
        finally:
            if downloaded_path is not None:
                downloaded_path.unlink(missing_ok=True)

    async def _answer_internal_task(self, request: OpenWebUIRequest) -> str:
        log_event("request.internal_task", task=request.internal_task)
        return await self.llm_client.generate_text(
            system_prompt=(
                "Выполни внутреннюю служебную задачу OpenWebUI "
                "строго по инструкции пользователя."
            ),
            user_prompt=request.user_message,
            temperature=0.0,
        )

    def _progress(
        self,
        callback: Callable[[str], None] | None,
        event: str,
    ) -> None:
        if callback is not None:
            callback(self.formatter.progress(event))

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
