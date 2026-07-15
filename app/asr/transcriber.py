from pathlib import Path
from threading import Event

from faster_whisper import WhisperModel

from app.schemas import RawTranscriptSegment, WordTimestamp


class TranscriptionError(RuntimeError):
    pass


class TranscriptionCancelled(TranscriptionError):
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
        cancel_event: Event | None = None,
    ) -> list[
        RawTranscriptSegment]:
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

            result: list[RawTranscriptSegment] = []

            # Inference actually starts during iteration.
            for segment in segments:
                if (
                    cancel_event is not None
                    and cancel_event.is_set()
                ):
                    raise TranscriptionCancelled(
                        "Transcription was cancelled."
                    )

                segment_text = segment.text.strip()
                words: list[WordTimestamp] = []

                for word in segment.words or []:
                    word_text = word.word.strip()

                    if not word_text:
                        continue

                    if word.start is None or word.end is None:
                        continue

                    words.append(
                        WordTimestamp(
                            start=round(float(word.start), 3),
                            end=round(float(word.end), 3),
                            word=word_text,
                        )
                    )

                if not segment_text and not words:
                    continue

                result.append(
                    RawTranscriptSegment(
                        start=round(float(segment.start), 3),
                        end=round(float(segment.end), 3),
                        text=segment_text,
                        words=words,
                    )
                )

            return result

        except TranscriptionCancelled:
            raise

        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed: {exc}"
            ) from exc

    def unload(self) -> None:
        self.model = None
