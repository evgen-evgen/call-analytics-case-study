"""
title: MTBank ASR
author: Evgeni Basov
date: 2026-07-11
version: 0.2.0
license: MIT
description: OpenWebUI adapter for MTBank call analysis.
"""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, Field

from app.config import AppSettings
from app.agents.analysis_chat import AnalysisChatAgent
from app.asr import AudioSourceError, OpenWebUIAudioDownloader
from app.llm.client import LLMClient
from app.metrics import start_metrics_server
from app.observability import configure_logging, log_event
from app.openwebui import (
    OpenWebUIRequestHandler,
    OpenWebUIRequestParser,
    OpenWebUIResponseFormatter,
    SyncStreamBridge,
)
from app.services import AudioAnalysisService, build_audio_analysis_service


@dataclass(frozen=True)
class PipelineRuntime:
    llm_client: LLMClient
    analysis_service: AudioAnalysisService
    handler: OpenWebUIRequestHandler
    bridge: SyncStreamBridge


class Pipeline:
    class Valves(BaseModel):
        WHISPER_MODEL: str = "medium"
        WHISPER_DEVICE: str = "cpu"
        WHISPER_COMPUTE_TYPE: str = "int8"
        WHISPER_LANGUAGE: str = "ru"
        OPENWEBUI_BASE_URL: str = "http://open-webui:8080"
        DIARIZATION_MODEL: str = (
            "pyannote/speaker-diarization-community-1"
        )
        DIARIZATION_DEVICE: str = "cpu"
        MAX_CONCURRENT_AUDIO_JOBS: int = Field(default=1, ge=1)

        @classmethod
        def from_settings(cls, settings: AppSettings) -> Self:
            return cls(
                WHISPER_MODEL=settings.asr.whisper_model,
                WHISPER_DEVICE=settings.asr.whisper_device,
                WHISPER_COMPUTE_TYPE=settings.asr.whisper_compute_type,
                WHISPER_LANGUAGE=settings.asr.whisper_language,
                OPENWEBUI_BASE_URL=settings.openwebui.base_url,
                DIARIZATION_MODEL=settings.asr.diarization_model,
                DIARIZATION_DEVICE=settings.asr.diarization_device,
                MAX_CONCURRENT_AUDIO_JOBS=(
                    settings.runtime.max_concurrent_audio_jobs
                ),
            )

    def __init__(self) -> None:
        self.settings = AppSettings.from_env()
        configure_logging(self.settings.runtime)
        self.id = "mtbank-asr"
        self.name = "MTBank ASR"
        self.valves = self.Valves.from_settings(self.settings)
        self.parser = OpenWebUIRequestParser()
        self.formatter = OpenWebUIResponseFormatter()
        self.runtime: PipelineRuntime | None = None

    async def on_startup(self) -> None:
        start_metrics_server(self.settings.runtime.metrics_port)
        llm_client = LLMClient(settings=self.settings.llm)
        try:
            self.runtime = self._build_runtime(
                llm_client=llm_client,
                event_loop=asyncio.get_running_loop(),
            )
        except Exception:
            await llm_client.client.close()
            raise
        log_event("pipeline.started")

    async def on_shutdown(self) -> None:
        runtime = self.runtime
        self.runtime = None
        if runtime is not None:
            runtime.analysis_service.close()
            await runtime.llm_client.client.close()
        log_event("pipeline.stopped")

    async def on_valves_updated(self) -> None:
        current = self.runtime
        if current is None:
            raise RuntimeError("Pipeline runtime is not initialized.")
        replacement = self._build_runtime(
            llm_client=current.llm_client,
            event_loop=current.bridge.event_loop,
        )
        self.runtime = replacement
        current.analysis_service.close()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> str | Iterator[str]:
        runtime = self.runtime
        if runtime is None:
            return "Pipeline не был корректно инициализирован."

        try:
            request = self.parser.parse(
                user_message=user_message,
                model_id=model_id,
                messages=messages,
                body=body,
            )
        except AudioSourceError as exc:
            return self.formatter.error(exc)

        if request.stream:
            return runtime.bridge.stream(
                lambda progress, cancel: runtime.handler.handle(
                    request,
                    progress=progress,
                    cancel_event=cancel,
                )
            )
        return runtime.bridge.run(runtime.handler.handle(request))

    def _build_runtime(
        self,
        *,
        llm_client: LLMClient,
        event_loop: asyncio.AbstractEventLoop,
    ) -> PipelineRuntime:
        analysis_service = build_audio_analysis_service(
            llm_client=llm_client,
            whisper_model=self.valves.WHISPER_MODEL,
            whisper_device=self.valves.WHISPER_DEVICE,
            whisper_compute_type=self.valves.WHISPER_COMPUTE_TYPE,
            whisper_language=self.valves.WHISPER_LANGUAGE.strip() or None,
            diarization_model=self.valves.DIARIZATION_MODEL,
            diarization_device=self.valves.DIARIZATION_DEVICE,
            hf_token=self.settings.asr.hf_token,
        )
        try:
            handler = OpenWebUIRequestHandler(
                downloader=OpenWebUIAudioDownloader(
                    base_url=self.valves.OPENWEBUI_BASE_URL,
                    api_key=self.settings.openwebui.api_key,
                ),
                analysis_service=analysis_service,
                chat_agent=AnalysisChatAgent(
                    llm_client,
                    max_context_chars=(
                        self.settings.runtime.chat_context_max_chars
                    ),
                ),
                llm_client=llm_client,
                formatter=self.formatter,
                max_concurrent_audio_jobs=(
                    self.valves.MAX_CONCURRENT_AUDIO_JOBS
                ),
            )
        except Exception:
            analysis_service.close()
            raise
        return PipelineRuntime(
            llm_client=llm_client,
            analysis_service=analysis_service,
            handler=handler,
            bridge=SyncStreamBridge(event_loop),
        )
