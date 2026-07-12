from pathlib import Path

from faster_whisper import WhisperModel

from pipelines.app.schemas import TranscriptSegment


class TranscriptionError(RuntimeError):
    pass


class Transcriber:
    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "ru",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language

        self.model: WhisperModel | None = None

    def load(self) -> None:
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def run(
        self,
        audio_path: Path,
    ) -> list[TranscriptSegment]:
        if self.model is None:
            raise TranscriptionError(
                "Whisper model is not loaded."
            )

        if not audio_path.exists():
            raise TranscriptionError(
                f"Audio file does not exist: {audio_path}"
            )

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=True,
            )

            result: list[TranscriptSegment] = []

            # Inference actually starts during iteration.
            for segment in segments:
                text = segment.text.strip()

                if not text:
                    continue

                result.append(
                    TranscriptSegment(
                        start=round(float(segment.start), 2),
                        end=round(float(segment.end), 2),
                        text=text,
                    )
                )

            return result

        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed: {exc}"
            ) from exc

    def unload(self) -> None:
        self.model = None
