from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from app.api.audio_source import ApiAudioSourceError, ApiAudioSourceService
from app.api.jobs import AnalysisJobManager
from app.config import AppSettings
from app.llm.client import LLMClient
from app.metrics import start_metrics_server
from app.observability import (
    configure_logging,
    log_event,
)
from app.schemas import AnalysisJobResponse, AnalyzeAccepted
from app.services import AudioAnalysisService, build_audio_analysis_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = AppSettings.from_env()
    configure_logging(settings.runtime)
    start_metrics_server(settings.runtime.metrics_port)
    llm_client = LLMClient(settings=settings.llm)
    audio_source_service = ApiAudioSourceService(
        max_audio_bytes=settings.api.max_audio_bytes
    )
    service: AudioAnalysisService | None = None
    job_manager: AnalysisJobManager | None = None
    try:
        service = build_audio_analysis_service(
            llm_client=llm_client,
            whisper_model=settings.asr.whisper_model,
            whisper_device=settings.asr.whisper_device,
            whisper_compute_type=settings.asr.whisper_compute_type,
            whisper_language=settings.asr.whisper_language.strip() or None,
            diarization_model=settings.asr.diarization_model,
            diarization_device=settings.asr.diarization_device,
            hf_token=settings.asr.hf_token,
        )
        app.state.analysis_service = service
        app.state.audio_source_service = audio_source_service
        job_manager = AnalysisJobManager(
            analysis_service=service,
            max_concurrent_jobs=settings.runtime.max_concurrent_audio_jobs,
            retention_seconds=settings.api.job_retention_seconds,
        )
        app.state.analysis_jobs = job_manager
        log_event("api.started")
        yield
    finally:
        if job_manager is not None:
            await job_manager.close()
        if service is not None:
            service.close()
        await audio_source_service.close()
        await llm_client.client.close()
        log_event("api.stopped")


app = FastAPI(
    title="MTBank Call Analysis API",
    version="1.0.0",
    description="ASR and multi-agent call analysis service.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok" if hasattr(request.app.state, "analysis_service") else "starting",
        "service": "analysis-api",
    }


@app.post(
    "/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnalyzeAccepted,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "url": {"type": "string", "format": "uri"},
                        },
                    }
                },
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["url"],
                        "properties": {
                            "url": {"type": "string", "format": "uri"},
                        },
                    }
                },
            },
        }
    },
)
async def analyze(
    request: Request,
    response: Response,
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
) -> AnalyzeAccepted:
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        if isinstance(body, dict) and body.get("url") is not None:
            url = str(body["url"])

    source_path: Path | None = None
    try:
        source_service: ApiAudioSourceService = (
            request.app.state.audio_source_service
        )
        source_path = await source_service.stage(file=file, url=url)
        jobs: AnalysisJobManager = request.app.state.analysis_jobs
        job = jobs.submit(source_path)
        source_path = None
        status_url = str(request.url_for("get_analysis", job_id=job.job_id))
        response.headers["Location"] = status_url
        return AnalyzeAccepted(
            job_id=job.job_id,
            status=job.status,
            status_url=status_url,
        )
    except ApiAudioSourceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        if source_path is not None:
            source_path.unlink(missing_ok=True)


@app.get(
    "/analyses/{job_id}",
    response_model=AnalysisJobResponse,
    name="get_analysis",
)
async def get_analysis(request: Request, job_id: str) -> AnalysisJobResponse:
    jobs: AnalysisJobManager = request.app.state.analysis_jobs
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return job
