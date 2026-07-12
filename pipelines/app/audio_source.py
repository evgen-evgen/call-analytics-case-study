import os
import re
import tempfile
from pathlib import Path

import httpx

from pipelines.app.schemas import AudioSource


FILE_TAG_PATTERN = re.compile(
    r"<file\s+(?P<attributes>[^>]+)/>",
    re.IGNORECASE,
)

ATTRIBUTE_PATTERN = re.compile(
    r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>[^"]*)"'
)

SUPPORTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/vnd.wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
}

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
}


class AudioSourceError(ValueError):
    pass


def extract_audio_source(user_message: str) -> AudioSource:
    match = FILE_TAG_PATTERN.search(user_message)

    if match is None:
        raise AudioSourceError(
            "В сообщении не найден прикреплённый аудиофайл."
        )

    attributes = dict(
        ATTRIBUTE_PATTERN.findall(
            match.group("attributes")
        )
    )

    file_type = attributes.get("type")
    file_id = attributes.get("url")
    content_type = attributes.get("content_type")
    filename = attributes.get("name")

    if not file_id or not content_type or not filename:
        raise AudioSourceError(
            "OpenWebUI передал неполные метаданные файла."
        )

    if file_type != "file":
        raise AudioSourceError(
            f"Неподдерживаемый тип вложения: {file_type}"
        )

    extension = Path(filename).suffix.lower()

    if (
        content_type not in SUPPORTED_CONTENT_TYPES
        and extension not in SUPPORTED_EXTENSIONS
    ):
        raise AudioSourceError(
            "Неподдерживаемый формат. "
            "Поддерживаются WAV, MP3 и OGG."
        )

    return AudioSource(
        file_id=file_id,
        filename=filename,
        content_type=content_type,
    )


class OpenWebUIAudioDownloader:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv(
                "OPENWEBUI_BASE_URL",
                "http://open-webui:8080",
            )
        ).rstrip("/")

        self.api_key = api_key or os.getenv("OPENWEBUI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "OPENWEBUI_API_KEY is not configured."
            )

        self.client = httpx.Client(
            timeout=httpx.Timeout(120.0),
            follow_redirects=True,
        )

    def download(
        self,
        source: AudioSource,
    ) -> Path:
        url = (
            f"{self.base_url}/api/v1/files/"
            f"{source.file_id}/content"
        )

        response = self.client.get(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        response.raise_for_status()

        suffix = Path(source.filename).suffix.lower()

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(response.content)
            temporary_path = Path(temporary_file.name)

        if temporary_path.stat().st_size == 0:
            temporary_path.unlink(missing_ok=True)

            raise AudioSourceError(
                "OpenWebUI вернул пустой файл."
            )

        return temporary_path

    def close(self) -> None:
        self.client.close()
