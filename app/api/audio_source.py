from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import httpx

from app.observability import operation
from app.config import AppSettings


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg"}


class UploadedAudio(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


class ApiAudioSourceError(ValueError):
    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiAudioSourceService:
    """Stages REST uploads and remote URLs as temporary local audio files."""

    def __init__(
        self,
        *,
        max_audio_bytes: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.max_audio_bytes = (
            max_audio_bytes
            if max_audio_bytes is not None
            else AppSettings.from_env().api.max_audio_bytes
        )
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(120),
            follow_redirects=True,
        )

    async def stage(
        self,
        *,
        file: UploadedAudio | None,
        url: str | None,
    ) -> Path:
        if (file is None) == (url is None):
            raise ApiAudioSourceError(
                "Provide exactly one audio source: file or url.",
                status_code=400,
            )
        if file is not None:
            with operation("audio.upload"):
                return await self._save_upload(file)
        with operation("audio.download"):
            return await self._download_url(url or "")

    async def _save_upload(self, file: UploadedAudio) -> Path:
        suffix = self._validated_suffix(Path(file.filename or "").suffix.lower())
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            path = Path(temporary.name)
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > self.max_audio_bytes:
                    path.unlink(missing_ok=True)
                    raise ApiAudioSourceError(
                        "Audio file is too large.",
                        status_code=413,
                    )
                temporary.write(chunk)
        if size == 0:
            path.unlink(missing_ok=True)
            raise ApiAudioSourceError("Audio file is empty.")
        return path

    async def _download_url(self, url: str) -> Path:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ApiAudioSourceError("Invalid HTTP(S) audio URL.")
        suffix = self._validated_suffix(Path(parsed.path).suffix.lower())
        path: Path | None = None
        try:
            async with self.client.stream("GET", url) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                    path = Path(temporary.name)
                    size = 0
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > self.max_audio_bytes:
                            raise ApiAudioSourceError(
                                "Audio file is too large.",
                                status_code=413,
                            )
                        temporary.write(chunk)
            if size == 0:
                raise ApiAudioSourceError("Audio file is empty.")
            return path
        except ApiAudioSourceError:
            if path is not None:
                path.unlink(missing_ok=True)
            raise
        except httpx.HTTPError as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            raise ApiAudioSourceError("Unable to download audio URL.") from exc

    @staticmethod
    def _validated_suffix(suffix: str) -> str:
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            raise ApiAudioSourceError(
                "Supported audio formats: WAV, MP3 and OGG.",
                status_code=415,
            )
        return suffix

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
