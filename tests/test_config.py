import pytest
from pydantic import ValidationError

from app.config import AppSettings


def test_settings_load_and_type_environment_values() -> None:
    settings = AppSettings.from_env(
        {
            "WHISPER_MODEL": "small",
            "LLM_TIMEOUT_SECONDS": "45.5",
            "MAX_CONCURRENT_AUDIO_JOBS": "3",
            "API_MAX_AUDIO_BYTES": "2048",
            "ANALYSIS_JOB_RETENTION_SECONDS": "120",
            "SERVICE_NAME": "mtbank-analysis-api",
        }
    )

    assert settings.asr.whisper_model == "small"
    assert settings.llm.timeout_seconds == 45.5
    assert settings.runtime.max_concurrent_audio_jobs == 3
    assert settings.api.max_audio_bytes == 2048
    assert settings.api.job_retention_seconds == 120
    assert settings.runtime.service_name == "mtbank-analysis-api"


def test_settings_reject_invalid_environment_values() -> None:
    with pytest.raises(ValidationError):
        AppSettings.from_env({"MAX_CONCURRENT_AUDIO_JOBS": "0"})


def test_settings_have_one_canonical_llm_timeout_default() -> None:
    assert AppSettings.from_env({}).llm.timeout_seconds == 30
