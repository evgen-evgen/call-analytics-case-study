import httpx
import pytest

from app.api.audio_source import ApiAudioSourceError, ApiAudioSourceService


class StubUpload:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self.content = content
        self.offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.content):
            return b""
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


@pytest.mark.asyncio
async def test_stages_upload_and_leaves_cleanup_to_caller() -> None:
    client = httpx.AsyncClient()
    service = ApiAudioSourceService(client=client)
    path = await service.stage(
        file=StubUpload("call.wav", b"audio"),
        url=None,
    )
    try:
        assert path.suffix == ".wav"
        assert path.read_bytes() == b"audio"
    finally:
        path.unlink(missing_ok=True)
        await client.aclose()


@pytest.mark.asyncio
async def test_rejects_ambiguous_audio_source() -> None:
    client = httpx.AsyncClient()
    service = ApiAudioSourceService(client=client)
    try:
        with pytest.raises(ApiAudioSourceError) as error:
            await service.stage(file=None, url=None)
        assert error.value.status_code == 400
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_rejects_unsupported_upload_without_temp_file() -> None:
    client = httpx.AsyncClient()
    service = ApiAudioSourceService(client=client)
    try:
        with pytest.raises(ApiAudioSourceError) as error:
            await service.stage(
                file=StubUpload("call.txt", b"not audio"),
                url=None,
            )
        assert error.value.status_code == 415
    finally:
        await client.aclose()
