from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline

from app.schemas import DiarizationSegment


class DiarizationError(RuntimeError):
    """Raised when speaker diarization fails."""


class Diarizer:
    def __init__(
        self,
        model_name: str = (
            "pyannote/speaker-diarization-community-1"
        ),
        device: str = "cpu",
        hf_token: str | None = None,
        num_speakers: int = 2,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.num_speakers = num_speakers

        self.pipeline: Pipeline | None = None

    def load(self) -> None:
        if not self.hf_token:
            raise DiarizationError(
                "HF_TOKEN is not configured."
            )

        try:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token,
            )

            if self.pipeline is None:
                raise DiarizationError(
                    "Pyannote returned an empty pipeline. "
                    "Check access to the model."
                )

            if self.device == "cuda":
                if not torch.cuda.is_available():
                    raise DiarizationError(
                        "DIARIZATION_DEVICE=cuda, but CUDA "
                        "is not available."
                    )

                self.pipeline.to(torch.device("cuda"))

            elif self.device != "cpu":
                raise DiarizationError(
                    "Supported diarization devices: cpu, cuda."
                )

        except DiarizationError:
            raise

        except Exception as exc:
            raise DiarizationError(
                f"Failed to load diarization model: {exc}"
            ) from exc

    def run(
        self,
        audio_path: Path,
    ) -> list[DiarizationSegment]:
        if self.pipeline is None:
            raise DiarizationError(
                "Diarization pipeline is not loaded."
            )

        if not audio_path.exists():
            raise DiarizationError(
                f"Audio file does not exist: {audio_path}"
            )

        try:
            output = self.pipeline(
                str(audio_path),
                num_speakers=self.num_speakers,
            )

            annotation = self._extract_annotation(output)

            result: list[DiarizationSegment] = []

            for turn, _, speaker in annotation.itertracks(
                yield_label=True
            ):
                if turn.end <= turn.start:
                    continue

                result.append(
                    DiarizationSegment(
                        speaker=str(speaker),
                        start=round(float(turn.start), 2),
                        end=round(float(turn.end), 2),
                    )
                )

            return result

        except Exception as exc:
            raise DiarizationError(
                f"Speaker diarization failed: {exc}"
            ) from exc

    def _extract_annotation(self, output: Any) -> Any:
        """
        Community-1 may return a result object containing both
        regular and exclusive diarization.

        Exclusive diarization is convenient for later alignment
        with transcription because only one speaker is active at
        each point in its timeline.
        """

        exclusive = getattr(
            output,
            "exclusive_speaker_diarization",
            None,
        )

        if exclusive is not None:
            return exclusive

        regular = getattr(
            output,
            "speaker_diarization",
            None,
        )

        if regular is not None:
            return regular

        # Compatibility fallback for versions returning Annotation
        # directly.
        if hasattr(output, "itertracks"):
            return output

        raise DiarizationError(
            "Unsupported pyannote output format."
        )

    def unload(self) -> None:
        self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
