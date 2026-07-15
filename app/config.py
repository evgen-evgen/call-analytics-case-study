from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, Field


class ASRSettings(BaseModel):
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "ru"
    hf_token: str | None = None
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_device: str = "cpu"


class LLMSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "analysis-model"
    timeout_seconds: float = Field(default=30, gt=0)
    max_completion_tokens: int = Field(default=8192, ge=1)
    max_concurrent_requests: int = Field(default=2, ge=1)
    rate_limit_max_retries: int = Field(default=3, ge=0)
    rate_limit_max_delay_seconds: float = Field(default=60, ge=0)


class RuntimeSettings(BaseModel):
    service_name: str = "mtbank-pipelines"
    max_concurrent_audio_jobs: int = Field(default=1, ge=1)
    chat_context_max_chars: int = Field(default=40_000, ge=1)
    metrics_port: int = Field(default=9100, ge=1, le=65_535)
    log_level: str = "INFO"
    runtime_log_level: str = "WARNING"


class APISettings(BaseModel):
    max_audio_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    job_retention_seconds: int = Field(default=3600, ge=1)


class OpenWebUISettings(BaseModel):
    base_url: str = "http://open-webui:8080"
    api_key: str | None = None


class AppSettings(BaseModel):
    asr: ASRSettings = Field(default_factory=ASRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    api: APISettings = Field(default_factory=APISettings)
    openwebui: OpenWebUISettings = Field(default_factory=OpenWebUISettings)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> AppSettings:
        env = os.environ if environ is None else environ

        def values(mapping: Mapping[str, str]) -> dict[str, str]:
            return {
                field: env[name]
                for name, field in mapping.items()
                if name in env
            }

        return cls(
            asr=values(
                {
                    "WHISPER_MODEL": "whisper_model",
                    "WHISPER_DEVICE": "whisper_device",
                    "WHISPER_COMPUTE_TYPE": "whisper_compute_type",
                    "WHISPER_LANGUAGE": "whisper_language",
                    "HF_TOKEN": "hf_token",
                    "DIARIZATION_MODEL": "diarization_model",
                    "DIARIZATION_DEVICE": "diarization_device",
                }
            ),
            llm=values(
                {
                    "LLM_API_KEY": "api_key",
                    "LLM_BASE_URL": "base_url",
                    "LLM_MODEL": "model",
                    "LLM_TIMEOUT_SECONDS": "timeout_seconds",
                    "LLM_MAX_COMPLETION_TOKENS": "max_completion_tokens",
                    "LLM_MAX_CONCURRENT_REQUESTS": "max_concurrent_requests",
                    "LLM_RATE_LIMIT_MAX_RETRIES": "rate_limit_max_retries",
                    "LLM_RATE_LIMIT_MAX_DELAY_SECONDS": (
                        "rate_limit_max_delay_seconds"
                    ),
                }
            ),
            runtime=values(
                {
                    "SERVICE_NAME": "service_name",
                    "MAX_CONCURRENT_AUDIO_JOBS": "max_concurrent_audio_jobs",
                    "CHAT_CONTEXT_MAX_CHARS": "chat_context_max_chars",
                    "METRICS_PORT": "metrics_port",
                    "LOG_LEVEL": "log_level",
                    "RUNTIME_LOG_LEVEL": "runtime_log_level",
                }
            ),
            api=values(
                {
                    "API_MAX_AUDIO_BYTES": "max_audio_bytes",
                    "ANALYSIS_JOB_RETENTION_SECONDS": "job_retention_seconds",
                }
            ),
            openwebui=values(
                {
                    "OPENWEBUI_BASE_URL": "base_url",
                    "OPENWEBUI_API_KEY": "api_key",
                }
            ),
        )
