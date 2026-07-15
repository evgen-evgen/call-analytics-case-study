
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from app.observability import log_event, operation
from app.metrics import CallAnalyticsMetrics, metrics

if TYPE_CHECKING:
    from app.agents.supervisor import AnalysisResult, AnalysisSupervisor
    from app.llm.client import LLMClient
    from app.asr import (
        AudioNormalizer,
        Diarizer,
        SpeakerRoleMapper,
        Transcriber,
        TranscriptAligner,
    )


class AudioAnalysisError(RuntimeError):
    """Raised when the complete audio analysis fails."""


class AudioAnalysisService:
    def __init__(
        self,
        *,
        normalizer: AudioNormalizer,
        transcriber: Transcriber,
        diarizer: Diarizer,
        aligner: TranscriptAligner,
        role_mapper: SpeakerRoleMapper,
        supervisor: AnalysisSupervisor,
        metrics_recorder: CallAnalyticsMetrics = metrics,
    ) -> None:
        self.normalizer = normalizer
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.aligner = aligner
        self.role_mapper = role_mapper
        self.supervisor = supervisor
        self.metrics = metrics_recorder

    async def analyze_path(
        self,
        audio_path: Path,
        *,
        progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AnalysisResult:
        if not audio_path.exists():
            raise AudioAnalysisError(
                f"Audio file does not exist: {audio_path}"
            )

        normalized_path: Path | None = None

        try:
            self._raise_if_cancelled(cancel_event)
            with operation("audio.normalize"):
                normalized_path = await asyncio.to_thread(
                    self.normalizer.normalize,
                    audio_path,
                )
            self._report(progress, "audio_prepared")
            self._raise_if_cancelled(cancel_event)

            with operation("audio.transcribe"):
                raw_transcript = await asyncio.to_thread(
                    self.transcriber.run,
                    normalized_path,
                    cancel_event,
                )
            self._report(progress, "transcription_completed")
            log_event(
                "transcript.created",
                segment_count=len(raw_transcript),
                word_count=sum(
                    len(segment.words) for segment in raw_transcript
                ),
            )
            self._raise_if_cancelled(cancel_event)

            with operation("audio.diarize"):
                diarization = await asyncio.to_thread(
                    self.diarizer.run,
                    normalized_path,
                )
            self._report(progress, "diarization_completed")
            log_event(
                "diarization.created",
                segment_count=len(diarization),
                speaker_count=len({segment.speaker for segment in diarization}),
            )
            self._raise_if_cancelled(cancel_event)

            with operation("transcript.align"):
                aligned_transcript = await asyncio.to_thread(
                    self.aligner.align,
                    transcript=raw_transcript,
                    diarization=diarization,
                )

            with operation("transcript.map_roles"):
                transcript = await asyncio.to_thread(
                    self.role_mapper.map_roles,
                    aligned_transcript,
                )

            if not transcript:
                raise AudioAnalysisError(
                    "Audio processing produced an empty transcript."
                )

            self._report(progress, "analysis_started")
            self._raise_if_cancelled(cancel_event)
            with operation("agents.analyze"):
                result = await self.supervisor.run(transcript)
            self.metrics.record_analysis(result)
            self._report(progress, "analysis_completed")
            return result

        finally:
            if normalized_path is not None:
                normalized_path.unlink(missing_ok=True)

    @staticmethod
    def _report(
        callback: Callable[[str], None] | None,
        event: str,
    ) -> None:
        if callback is not None:
            callback(event)

    @staticmethod
    def _raise_if_cancelled(
        cancel_event: threading.Event | None,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError

    def close(self) -> None:
        self.transcriber.unload()
        self.diarizer.unload()


def build_audio_analysis_service(
    *,
    llm_client: LLMClient,
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    whisper_language: str | None,
    diarization_model: str,
    diarization_device: str,
    hf_token: str | None,
) -> AudioAnalysisService:
    from app.agents.supervisor import AnalysisSupervisor
    from app.asr import (
        AudioNormalizer,
        Diarizer,
        SpeakerRoleMapper,
        Transcriber,
        TranscriptAligner,
    )

    transcriber = Transcriber(
        model_name=whisper_model,
        device=whisper_device,
        compute_type=whisper_compute_type,
        language=whisper_language,
    )
    diarizer = Diarizer(
        model_name=diarization_model,
        device=diarization_device,
        hf_token=hf_token,
        num_speakers=2,
    )

    try:
        with operation(
            "model.load.whisper",
            model=whisper_model,
            device=whisper_device,
        ):
            transcriber.load()

        with operation(
            "model.load.diarization",
            model=diarization_model,
            device=diarization_device,
        ):
            diarizer.load()
    except Exception:
        transcriber.unload()
        diarizer.unload()
        raise

    return AudioAnalysisService(
        normalizer=AudioNormalizer(sample_rate=16_000, channels=1),
        transcriber=transcriber,
        diarizer=diarizer,
        aligner=TranscriptAligner(merge_gap_seconds=1.0),
        role_mapper=SpeakerRoleMapper(),
        supervisor=AnalysisSupervisor(llm_client),
    )
