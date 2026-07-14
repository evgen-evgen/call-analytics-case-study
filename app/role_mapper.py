from .schemas import TranscriptSegment


class SpeakerRoleMapper:
    """
    Baseline assumption for a two-party call-center conversation:

    - first detected speaker -> Оператор
    - second detected speaker -> Клиент
    """

    def map_roles(
        self,
        transcript: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        if not transcript:
            return []

        ordered_segments = sorted(
            transcript,
            key=lambda segment: (
                segment.start,
                segment.end,
            ),
        )

        speakers: list[str] = []

        for segment in ordered_segments:
            if segment.speaker not in speakers:
                speakers.append(segment.speaker)

        role_map: dict[str, str] = {}

        if speakers:
            role_map[speakers[0]] = "Оператор"

        if len(speakers) >= 2:
            role_map[speakers[1]] = "Клиент"

        for index, speaker in enumerate(
            speakers[2:],
            start=3,
        ):
            role_map[speaker] = f"Спикер {index}"

        return [
            TranscriptSegment(
                speaker=role_map.get(
                    segment.speaker,
                    segment.speaker,
                ),
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
            for segment in ordered_segments
        ]