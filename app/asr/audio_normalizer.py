import subprocess
import tempfile
from pathlib import Path


class AudioNormalizationError(RuntimeError):
    """Raised when an audio file cannot be normalized."""


class AudioNormalizer:
    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def normalize(
        self,
        source_path: Path,
    ) -> Path:
        if not source_path.exists():
            raise AudioNormalizationError(
                f"Source audio does not exist: {source_path}"
            )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            normalized_path = Path(temporary_file.name)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(normalized_path),
        ]

        try:
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )

            if completed_process.returncode != 0:
                raise AudioNormalizationError(
                    "FFmpeg failed to normalize audio: "
                    f"{completed_process.stderr.strip()}"
                )

            if not normalized_path.exists():
                raise AudioNormalizationError(
                    "FFmpeg did not produce an output file."
                )

            if normalized_path.stat().st_size == 0:
                raise AudioNormalizationError(
                    "FFmpeg produced an empty output file."
                )

            return normalized_path

        except subprocess.TimeoutExpired as exc:
            normalized_path.unlink(missing_ok=True)

            raise AudioNormalizationError(
                "Audio normalization timed out."
            ) from exc

        except AudioNormalizationError:
            normalized_path.unlink(missing_ok=True)
            raise

        except Exception as exc:
            normalized_path.unlink(missing_ok=True)

            raise AudioNormalizationError(
                f"Audio normalization failed: {exc}"
            ) from exc