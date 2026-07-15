#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

from jiwer import process_characters, process_words

from app.asr import AudioNormalizer, Transcriber
from app.config import AppSettings


AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".oga", ".m4a", ".flac"}
CASE_ID_PATTERN = re.compile(r"^(?:call_)?(\d{2})(?:_|$)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def read_reference(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    for line in lines:
        value = line.strip()
        if value.startswith('text = """'):
            value = value.removeprefix('text = """').strip()
        if value == '"""':
            continue
        value = re.sub(
            r"^(?:О|К|Оператор|Клиент)\s*:\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        if value:
            cleaned.append(value)
    return " ".join(cleaned)


def case_id(path: Path) -> str | None:
    match = CASE_ID_PATTERN.match(path.stem)
    return match.group(1) if match else None


def pair_cases(
    audio_dir: Path,
    reference_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    audio_files = sorted(
        path
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    references = {
        identifier: path
        for path in sorted(reference_dir.glob("*.txt"))
        if (identifier := case_id(path)) is not None
    }
    pairs = [
        (audio, references[identifier])
        for audio in audio_files
        if (identifier := case_id(audio)) in references
    ]
    matched_audio = {audio for audio, _ in pairs}
    return pairs, [audio for audio in audio_files if audio not in matched_audio]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate faster-whisper on test calls with jiwer.",
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("test_data"))
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("test_case/references"),
    )
    parser.add_argument(
        "--hypothesis-dir",
        type=Path,
        default=Path("test_case/hypotheses"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("test_case/reports/wer.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("test_case/reports/wer.md"),
    )
    return parser.parse_args()


def write_reports(rows: list[dict[str, object]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "audio",
        "reference_words",
        "hypothesis_words",
        "substitutions",
        "deletions",
        "insertions",
        "wer",
        "cer",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Запись | Слов в эталоне | S | D | I | WER | CER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| {audio} | {reference_words} | {substitutions} | {deletions} | "
        "{insertions} | {wer:.2%} | {cer:.2%} |".format(**row)
        for row in rows
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pairs, skipped_audio = pair_cases(args.audio_dir, args.reference_dir)
    if not pairs:
        raise SystemExit("No matching audio/reference pairs found")
    if skipped_audio:
        print(
            "Skipped without references: "
            + ", ".join(path.name for path in skipped_audio)
        )

    settings = AppSettings.from_env().asr
    transcriber = Transcriber(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
    )
    normalizer = AudioNormalizer()
    args.hypothesis_dir.mkdir(parents=True, exist_ok=True)
    transcriber.load()
    rows: list[dict[str, object]] = []
    all_references: list[str] = []
    all_hypotheses: list[str] = []

    try:
        for index, (audio_path, reference_path) in enumerate(pairs, start=1):
            print(
                f"[{index}/{len(pairs)}] Transcribing {audio_path.name}...",
                flush=True,
            )
            normalized_path = normalizer.normalize(audio_path)
            try:
                segments = transcriber.run(normalized_path)
            finally:
                normalized_path.unlink(missing_ok=True)

            hypothesis = " ".join(segment.text for segment in segments).strip()
            (args.hypothesis_dir / f"{audio_path.stem}.txt").write_text(
                hypothesis + "\n",
                encoding="utf-8",
            )
            reference = read_reference(reference_path)
            reference_normalized = normalize_text(reference)
            hypothesis_normalized = normalize_text(hypothesis)
            all_references.append(reference_normalized)
            all_hypotheses.append(hypothesis_normalized)
            word_result = process_words(reference_normalized, hypothesis_normalized)
            char_result = process_characters(
                reference_normalized,
                hypothesis_normalized,
            )
            rows.append(
                {
                    "audio": audio_path.name,
                    "reference_words": len(reference_normalized.split()),
                    "hypothesis_words": len(hypothesis_normalized.split()),
                    "substitutions": word_result.substitutions,
                    "deletions": word_result.deletions,
                    "insertions": word_result.insertions,
                    "wer": word_result.wer,
                    "cer": char_result.cer,
                }
            )
            print(
                f"[{index}/{len(pairs)}] {audio_path.name}: "
                f"WER={word_result.wer:.2%}, CER={char_result.cer:.2%}",
                flush=True,
            )
    finally:
        transcriber.unload()

    total_reference = " ".join(all_references)
    total_hypothesis = " ".join(all_hypotheses)
    total_words = process_words(total_reference, total_hypothesis)
    total_chars = process_characters(total_reference, total_hypothesis)
    rows.append(
        {
            "audio": "Итого",
            "reference_words": len(total_reference.split()),
            "hypothesis_words": len(total_hypothesis.split()),
            "substitutions": total_words.substitutions,
            "deletions": total_words.deletions,
            "insertions": total_words.insertions,
            "wer": total_words.wer,
            "cer": total_chars.cer,
        }
    )

    write_reports(rows, args.output_csv, args.output_md)
    print(f"Evaluated {len(pairs)} files")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
