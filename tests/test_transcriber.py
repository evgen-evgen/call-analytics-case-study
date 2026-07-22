from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from app.asr.transcriber import (
    Transcriber,
    TranscriptionCancelled,
    TranscriptionError,
)
from app.schemas import RawTranscriptSegment, WordTimestamp


class FakeWhisperModel:
    def __init__(self, segments=None, error: Exception | None = None) -> None:
        self.segments = segments or []
        self.error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(self, audio_path: str, **kwargs):
        self.calls.append((audio_path, kwargs))

        if self.error is not None:
            raise self.error

        return iter(self.segments), SimpleNamespace(language="ru")


def make_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"audio")
    return audio_path


def test_run_returns_typed_segments_with_word_timestamps(
    tmp_path: Path,
) -> None:
    audio_path = make_audio(tmp_path)
    model = FakeWhisperModel(
        segments=[
            SimpleNamespace(
                start=1.23456,
                end=3.45678,
                text="  Добрый день.  ",
                words=[
                    SimpleNamespace(
                        start=1.23456,
                        end=1.78954,
                        word=" Добрый",
                    ),
                    SimpleNamespace(
                        start=1.80049,
                        end=2.34551,
                        word=" день.",
                    ),
                ],
            )
        ]
    )
    transcriber = Transcriber(language="ru")
    transcriber.model = model

    result = transcriber.run(audio_path)

    assert result == [
        RawTranscriptSegment(
            start=1.235,
            end=3.457,
            text="Добрый день.",
            words=[
                WordTimestamp(start=1.235, end=1.79, word="Добрый"),
                WordTimestamp(start=1.8, end=2.346, word="день."),
            ],
        )
    ]
    assert model.calls == [
        (
            str(audio_path),
            {
                "language": "ru",
                "beam_size": 5,
                "vad_filter": True,
                "word_timestamps": True,
                "condition_on_previous_text": True,
            },
        )
    ]


def test_run_filters_invalid_words_and_keeps_segment_text(
    tmp_path: Path,
) -> None:
    audio_path = make_audio(tmp_path)
    model = FakeWhisperModel(
        segments=[
            SimpleNamespace(
                start=0.0,
                end=1.0,
                text=" Текст без пригодных timestamps ",
                words=[
                    SimpleNamespace(start=0.0, end=0.1, word="   "),
                    SimpleNamespace(start=None, end=0.5, word="Текст"),
                    SimpleNamespace(start=0.5, end=None, word="без"),
                ],
            ),
            SimpleNamespace(
                start=1.0,
                end=2.0,
                text=" Ещё текст ",
                words=None,
            ),
        ]
    )
    transcriber = Transcriber()
    transcriber.model = model

    result = transcriber.run(audio_path)

    assert [segment.text for segment in result] == [
        "Текст без пригодных timestamps",
        "Ещё текст",
    ]
    assert all(segment.words == [] for segment in result)


def test_run_skips_completely_empty_segment(tmp_path: Path) -> None:
    audio_path = make_audio(tmp_path)
    model = FakeWhisperModel(
        segments=[
            SimpleNamespace(
                start=0.0,
                end=1.0,
                text="   ",
                words=None,
            )
        ]
    )
    transcriber = Transcriber()
    transcriber.model = model

    assert transcriber.run(audio_path) == []


def test_run_requires_loaded_model(tmp_path: Path) -> None:
    audio_path = make_audio(tmp_path)

    with pytest.raises(TranscriptionError, match="model is not loaded"):
        Transcriber().run(audio_path)


def test_run_requires_existing_audio_file(tmp_path: Path) -> None:
    transcriber = Transcriber()
    transcriber.model = FakeWhisperModel()

    with pytest.raises(TranscriptionError, match="Audio file does not exist"):
        transcriber.run(tmp_path / "missing.wav")


def test_run_preserves_cancellation_error(tmp_path: Path) -> None:
    audio_path = make_audio(tmp_path)
    transcriber = Transcriber()
    transcriber.model = FakeWhisperModel(
        segments=[
            SimpleNamespace(
                start=0.0,
                end=1.0,
                text="Текст",
                words=None,
            )
        ]
    )
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(TranscriptionCancelled):
        transcriber.run(audio_path, cancel_event)


def test_run_wraps_whisper_error(tmp_path: Path) -> None:
    audio_path = make_audio(tmp_path)
    transcriber = Transcriber()
    transcriber.model = FakeWhisperModel(error=RuntimeError("model failed"))

    with pytest.raises(
        TranscriptionError,
        match="Transcription failed: model failed",
    ) as error:
        transcriber.run(audio_path)

    assert isinstance(error.value.__cause__, RuntimeError)


def test_unload_releases_model() -> None:
    transcriber = Transcriber()
    transcriber.model = FakeWhisperModel()

    transcriber.unload()

    assert transcriber.model is None
