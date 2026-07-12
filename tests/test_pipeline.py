from pathlib import Path

from pipeline import Pipeline


def test_pipeline_returns_structured_analysis(tmp_path):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"fake-audio-bytes")

    pipeline = Pipeline(whisper_model="base")
    result = pipeline.analyze(str(audio_path))

    assert result["transcript"][0]["speaker"] in {"Оператор", "Клиент"}
    assert result["classification"]["topic"] in {"кредиты", "карты", "переводы", "жалобы", "другое"}
    assert result["quality_score"]["total"] >= 0
    assert result["compliance"]["passed"] in {True, False}
