from dataclasses import dataclass

from app.schemas import (
    DiarizationSegment,
    RawTranscriptSegment,
    TranscriptSegment,
    WordTimestamp,
)


class AlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerWord:
    speaker: str
    start: float
    end: float
    text: str


class TranscriptAligner:
    def __init__(
        self,
        merge_gap_seconds: float = 1.0,
        minimum_overlap_seconds: float = 0.01,
    ) -> None:
        self.merge_gap_seconds = merge_gap_seconds
        self.minimum_overlap_seconds = minimum_overlap_seconds

    def align(
        self,
        transcript: list[RawTranscriptSegment],
        diarization: list[DiarizationSegment],
    ) -> list[TranscriptSegment]:
        if not transcript:
            return []

        if not diarization:
            raise AlignmentError(
                "Diarization returned no speaker segments."
            )

        words = self._flatten_words(transcript)

        if not words:
            return self._align_whisper_segments(
                transcript=transcript,
                diarization=diarization,
            )

        attributed_words = [
            SpeakerWord(
                speaker=self._find_speaker(
                    start=word.start,
                    end=word.end,
                    diarization=diarization,
                ),
                start=word.start,
                end=word.end,
                text=word.word,
            )
            for word in words
        ]

        return self._merge_words(attributed_words)

    def _flatten_words(
        self,
        transcript: list[RawTranscriptSegment],
    ) -> list[WordTimestamp]:
        words: list[WordTimestamp] = []

        for segment in transcript:
            words.extend(segment.words)

        return sorted(
            words,
            key=lambda word: (word.start, word.end),
        )

    def _find_speaker(
        self,
        start: float,
        end: float,
        diarization: list[DiarizationSegment],
    ) -> str:
        best_speaker: str | None = None
        best_overlap = 0.0

        for speaker_segment in diarization:
            overlap = self._calculate_overlap(
                first_start=start,
                first_end=end,
                second_start=speaker_segment.start,
                second_end=speaker_segment.end,
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_segment.speaker

        if (
            best_speaker is not None
            and best_overlap >= self.minimum_overlap_seconds
        ):
            return best_speaker

        midpoint = (start + end) / 2

        for speaker_segment in diarization:
            if speaker_segment.start <= midpoint <= speaker_segment.end:
                return speaker_segment.speaker

        return self._find_nearest_speaker(
            start=start,
            end=end,
            diarization=diarization,
        )

    @staticmethod
    def _calculate_overlap(
        first_start: float,
        first_end: float,
        second_start: float,
        second_end: float,
    ) -> float:
        return max(
            0.0,
            min(first_end, second_end)
            - max(first_start, second_start),
        )

    @staticmethod
    def _find_nearest_speaker(
        start: float,
        end: float,
        diarization: list[DiarizationSegment],
    ) -> str:
        midpoint = (start + end) / 2

        nearest_segment = min(
            diarization,
            key=lambda segment: min(
                abs(midpoint - segment.start),
                abs(midpoint - segment.end),
            ),
        )

        return nearest_segment.speaker

    def _merge_words(
        self,
        words: list[SpeakerWord],
    ) -> list[TranscriptSegment]:
        if not words:
            return []

        result: list[TranscriptSegment] = []

        current_speaker = words[0].speaker
        current_start = words[0].start
        current_end = words[0].end
        current_words = [words[0].text]

        for word in words[1:]:
            gap = word.start - current_end

            same_turn = (
                word.speaker == current_speaker
                and gap <= self.merge_gap_seconds
            )

            if same_turn:
                current_words.append(word.text)
                current_end = max(current_end, word.end)
                continue

            result.append(
                TranscriptSegment(
                    speaker=current_speaker,
                    start=round(current_start, 2),
                    end=round(current_end, 2),
                    text=self._join_words(current_words),
                )
            )

            current_speaker = word.speaker
            current_start = word.start
            current_end = word.end
            current_words = [word.text]

        result.append(
            TranscriptSegment(
                speaker=current_speaker,
                start=round(current_start, 2),
                end=round(current_end, 2),
                text=self._join_words(current_words),
            )
        )

        return result

    def _align_whisper_segments(
        self,
        transcript: list[RawTranscriptSegment],
        diarization: list[DiarizationSegment],
    ) -> list[TranscriptSegment]:
        """
        Fallback for cases where faster-whisper returns no word timestamps.
        """

        result: list[TranscriptSegment] = []

        for segment in transcript:
            if not segment.text.strip():
                continue

            speaker = self._find_speaker(
                start=segment.start,
                end=segment.end,
                diarization=diarization,
            )

            result.append(
                TranscriptSegment(
                    speaker=speaker,
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=segment.text.strip(),
                )
            )

        return self._merge_segments(result)

    def _merge_segments(
        self,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        if not segments:
            return []

        result: list[TranscriptSegment] = []
        current = segments[0]

        for segment in segments[1:]:
            gap = segment.start - current.end

            if (
                segment.speaker == current.speaker
                and gap <= self.merge_gap_seconds
            ):
                current = TranscriptSegment(
                    speaker=current.speaker,
                    start=current.start,
                    end=max(current.end, segment.end),
                    text=self._join_words(
                        [current.text, segment.text]
                    ),
                )
                continue

            result.append(current)
            current = segment

        result.append(current)

        return result

    @staticmethod
    def _join_words(words: list[str]) -> str:
        """
        Whisper word tokens may contain leading spaces.
        Since we strip them in Transcriber, reconstruct readable text.
        """

        text = " ".join(
            word.strip()
            for word in words
            if word.strip()
        )

        replacements = {
            " ,": ",",
            " .": ".",
            " !": "!",
            " ?": "?",
            " :": ":",
            " ;": ";",
            " )": ")",
            "( ": "(",
        }

        for source, target in replacements.items():
            text = text.replace(source, target)

        return text.strip()
