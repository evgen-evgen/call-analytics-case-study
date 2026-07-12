"""
title: MTBank ASR Debug
author: Evgeni Basov
date: 2026-07-11
version: 0.1.0
license: MIT
description: Diagnostic pipeline for inspecting OpenWebUI audio attachments.
requirements: faster-whisper>=1.2.1
"""
import os
import json
from pathlib import Path
from typing import Any, Generator, Iterator, Union

import httpx
from pydantic import BaseModel, Field

from pipelines.app.audio_source import (
    AudioSourceError,
    OpenWebUIAudioDownloader,
    extract_audio_source,
)

from pipelines.app.transcriber import (
    Transcriber,
    TranscriptionError,
)

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

    def __init__(self) -> None:
        self.name = "MTBank ASR"
        self.valves = self.Valves(
            WHISPER_MODEL=os.getenv("WHISPER_MODEL", "medium"),
            WHISPER_DEVICE=os.getenv("WHISPER_DEVICE", "cpu"),
            WHISPER_COMPUTE_TYPE=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            WHISPER_LANGUAGE=os.getenv("WHISPER_LANGUAGE", "ru"),
            OPENWEBUI_BASE_URL=os.getenv("OPENWEBUI_BASE_URL", "http://open-webui:8080"),
            DEBUG_OUTPUT_MAX_LENGTH=int(os.getenv("DEBUG_OUTPUT_MAX_LENGTH", "20000")),
        )
        self.downloader: OpenWebUIAudioDownloader | None = None
        self.transcriber: Transcriber | None = None

    async def on_startup(self) -> None:
        self.downloader = OpenWebUIAudioDownloader(
            base_url=self.valves.OPENWEBUI_BASE_URL,
        )

        self.transcriber = Transcriber(
            model_name=self.valves.WHISPER_MODEL,
            device=self.valves.WHISPER_DEVICE,
            compute_type=self.valves.WHISPER_COMPUTE_TYPE,
            language=self.valves.WHISPER_LANGUAGE,
        )
        print(
            "Loading faster-whisper model: "
            f"model={self.valves.WHISPER_MODEL}, "
            f"device={self.valves.WHISPER_DEVICE}, "
            f"compute_type={self.valves.WHISPER_COMPUTE_TYPE}"
        )

        # Loading is synchronous and can download model weights.
        # Startup happens once, so the model is not reloaded per request.
        self.transcriber.load()

        print("MTBank ASR pipeline started successfully.")

    async def on_shutdown(self) -> None:
        if self.transcriber is not None:
            self.transcriber.unload()

        print("MTBank ASR pipeline stopped.")


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

        temporary_path: Path | None = None

        try:
            source = extract_audio_source(user_message)
            temporary_path = self.downloader.download(source)
            size_bytes = temporary_path.stat().st_size
            transcript = self.transcriber.run(temporary_path)


            response = {
                "filename": source.filename,
                "content_type": source.content_type,
                "model": self.valves.WHISPER_MODEL,
                "segments": [
                    segment.model_dump()
                    for segment in transcript
                ],
            }

            formatted_json = json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
            )

            return (
                "## Транскрипт\n\n"
                f"```json\n{formatted_json}\n```"
            )

        except AudioSourceError as exc:
            return (
                "## Ошибка входного файла\n\n"
                f"{exc}"
            )

        except TranscriptionError as exc:
            return (
                "## Ошибка транскрибации\n\n"
                f"{exc}"
            )

        except httpx.HTTPStatusError as exc:
            return (
                "## Ошибка загрузки файла\n\n"
                f"OpenWebUI API status: "
                f"`{exc.response.status_code}`\n\n"
                f"Response: `{exc.response.text[:500]}`"
            )

        except httpx.RequestError as exc:
            return (
                "## Ошибка подключения к OpenWebUI\n\n"
                f"`{exc}`"
            )

        except Exception as exc:
            return (
                "## Неожиданная ошибка\n\n"
                f"Type: `{type(exc).__name__}`\n\n"
                f"Message: `{exc}`"
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
