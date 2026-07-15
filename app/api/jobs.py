from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

from app.mappers import AnalyzeResponseMapper
from app.observability import (
    bind_request_id,
    log_event,
    reset_request_id,
)
from app.schemas import AnalysisJobResponse, AnalysisJobStatus, AnalyzeResponse
from app.services import AudioAnalysisService


@dataclass
class _AnalysisJob:
    job_id: str
    status: AnalysisJobStatus = AnalysisJobStatus.QUEUED
    result: AnalyzeResponse | None = None
    error: str | None = None
    finished_at: float | None = None

    def response(self) -> AnalysisJobResponse:
        return AnalysisJobResponse(
            job_id=self.job_id,
            status=self.status,
            result=self.result,
            error=self.error,
        )


class AnalysisJobManager:
    def __init__(
        self,
        *,
        analysis_service: AudioAnalysisService,
        max_concurrent_jobs: int,
        retention_seconds: int,
        mapper: AnalyzeResponseMapper | None = None,
    ) -> None:
        self.analysis_service = analysis_service
        self.mapper = mapper or AnalyzeResponseMapper()
        self.retention_seconds = retention_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, _AnalysisJob] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, source_path: Path) -> AnalysisJobResponse:
        self._remove_expired_jobs()
        job = _AnalysisJob(job_id=str(uuid4()))
        self._jobs[job.job_id] = job
        task = asyncio.create_task(
            self._run(job, source_path),
            name=f"analysis-job-{job.job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        log_event("analysis_job.queued", job_id=job.job_id)
        return job.response()

    def get(self, job_id: str) -> AnalysisJobResponse | None:
        self._remove_expired_jobs()
        job = self._jobs.get(job_id)
        return job.response() if job is not None else None

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, job: _AnalysisJob, source_path: Path) -> None:
        token = bind_request_id(job.job_id)
        started_at = perf_counter()
        try:
            async with self._semaphore:
                job.status = AnalysisJobStatus.PROCESSING
                log_event("analysis_job.processing", job_id=job.job_id)
                result = await self.analysis_service.analyze_path(source_path)
                job.result = self.mapper.map(result)
                job.status = AnalysisJobStatus.COMPLETED
                log_event(
                    "analysis_job.completed",
                    job_id=job.job_id,
                    duration_ms=round((perf_counter() - started_at) * 1000, 2),
                )
        except asyncio.CancelledError:
            log_event("analysis_job.cancelled", job_id=job.job_id)
            raise
        except Exception as exc:
            job.status = AnalysisJobStatus.FAILED
            job.error = "Audio analysis failed."
            log_event(
                "analysis_job.failed",
                level=40,
                exc_info=True,
                job_id=job.job_id,
                error_type=type(exc).__name__,
                error_message="Background audio analysis failed",
            )
        finally:
            source_path.unlink(missing_ok=True)
            if job.status in {
                AnalysisJobStatus.COMPLETED,
                AnalysisJobStatus.FAILED,
            }:
                job.finished_at = monotonic()
            reset_request_id(token)

    def _remove_expired_jobs(self) -> None:
        now = monotonic()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None
            and now - job.finished_at >= self.retention_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]
