from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.audio_analysis import AudioAnalysisService


class FakeNormalizer:
    def __init__(self, normalized_path: Path) -> None:
        self.normalized_path = normalized_path

    def normalize(self, source_path: Path) -> Path:
        assert source_path.exists()
        return self.normalized_path


class FakeTranscriber:
    def run(self, audio_path: Path, cancel_event=None):
        assert audio_path.exists()
        return [SimpleNamespace(words=[SimpleNamespace(word="тест")])]


class FakeDiarizer:
    def run(self, audio_path: Path):
        assert audio_path.exists()
        return [SimpleNamespace(speaker="SPEAKER_00")]


class FakeAligner:
    def align(self, *, transcript, diarization):
        assert transcript
        assert diarization
        return ["aligned"]


class FakeRoleMapper:
    def map_roles(self, transcript):
        assert transcript == ["aligned"]
        return ["mapped"]


class FakeSupervisor:
    async def run(self, transcript):
        assert transcript == ["mapped"]
        return "analysis"


class FakeMetrics:
    def __init__(self) -> None:
        self.recorded = []

    def record_analysis(self, analysis) -> None:
        self.recorded.append(analysis)


@pytest.mark.asyncio
async def test_audio_analysis_service_runs_complete_use_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", run_inline)
    source_path = tmp_path / "source.mp3"
    source_path.write_bytes(b"source")
    normalized_path = tmp_path / "normalized.wav"
    normalized_path.write_bytes(b"normalized")
    progress: list[str] = []

    metrics = FakeMetrics()
    service = AudioAnalysisService(
        normalizer=FakeNormalizer(normalized_path),
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        aligner=FakeAligner(),
        role_mapper=FakeRoleMapper(),
        supervisor=FakeSupervisor(),
        metrics_recorder=metrics,
    )

    result = await service.analyze_path(source_path, progress=progress.append)

    assert result == "analysis"
    assert metrics.recorded == ["analysis"]
    assert progress[0] == "audio_prepared"
    assert set(progress[1:3]) == {
        "transcription_completed",
        "diarization_completed",
    }
    assert progress[3:] == ["analysis_started", "analysis_completed"]
    assert source_path.exists()
    assert not normalized_path.exists()
