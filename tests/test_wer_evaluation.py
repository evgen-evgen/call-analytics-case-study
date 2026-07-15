from scripts.evaluate_wer import (
    normalize_text,
    pair_cases,
    read_reference,
    write_reports,
)


def test_normalize_text_for_russian_wer() -> None:
    assert normalize_text("  Всё, хорошо!\nДа? ") == "все хорошо да"


def test_write_reports_creates_csv_and_markdown(tmp_path) -> None:
    rows = [
        {
            "audio": "call.wav",
            "reference_words": 10,
            "hypothesis_words": 9,
            "substitutions": 1,
            "deletions": 1,
            "insertions": 0,
            "wer": 0.2,
            "cer": 0.1,
        }
    ]
    csv_path = tmp_path / "wer.csv"
    md_path = tmp_path / "wer.md"

    write_reports(rows, csv_path, md_path)

    assert "call.wav,10,9,1,1,0,0.2,0.1" in csv_path.read_text()
    assert "| call.wav | 10 | 1 | 1 | 0 | 20.00% | 10.00% |" in (
        md_path.read_text()
    )


def test_read_reference_removes_roles_and_python_wrapper(tmp_path) -> None:
    reference = tmp_path / "call_01.txt"
    reference.write_text(
        'text = """\nО: Добрый день.\nК: Здравствуйте.\n"""',
        encoding="utf-8",
    )

    assert read_reference(reference) == "Добрый день. Здравствуйте."


def test_pair_cases_uses_numeric_prefix_and_skips_unmatched(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    reference_dir = tmp_path / "references"
    audio_dir.mkdir()
    reference_dir.mkdir()
    matching_audio = audio_dir / "01_credit.wav"
    unmatched_audio = audio_dir / "dialogue.mp3"
    matching_audio.touch()
    unmatched_audio.touch()
    reference = reference_dir / "call_01.txt"
    reference.touch()

    pairs, skipped = pair_cases(audio_dir, reference_dir)

    assert pairs == [(matching_audio, reference)]
    assert skipped == [unmatched_audio]
