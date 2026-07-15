import asyncio
from pathlib import Path
from app.api.jobs import AnalysisJobManager
from app.schemas import AnalysisJobStatus, AnalyzeResponse


class FakeMapper:
    def map(self, result):
        return result


def analysis_response() -> AnalyzeResponse:
    return AnalyzeResponse(
        transcript=[],
        classification={"topic": "cards", "priority": "low"},
        quality_score={
            "total": 100,
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": True,
                "farewell": True,
            },
        },
        compliance={"passed": True, "issues": []},
        summary="done",
        action_items=[],
    )


def test_analysis_job_moves_from_queue_to_completed(tmp_path: Path) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        expected = analysis_response()

        class Service:
            async def analyze_path(self, path: Path):
                assert path.exists()
                await release.wait()
                return expected

        source = tmp_path / "call.wav"
        source.write_bytes(b"audio")
        manager = AnalysisJobManager(
            analysis_service=Service(),
            max_concurrent_jobs=1,
            retention_seconds=3600,
            mapper=FakeMapper(),
        )

        accepted = manager.submit(source)
        assert accepted.status == AnalysisJobStatus.QUEUED
        await asyncio.sleep(0)
        assert manager.get(accepted.job_id).status == AnalysisJobStatus.PROCESSING

        release.set()
        await asyncio.sleep(0)
        completed = manager.get(accepted.job_id)
        assert completed.status == AnalysisJobStatus.COMPLETED
        assert completed.result is expected
        assert not source.exists()
        await manager.close()

    asyncio.run(scenario())


def test_analysis_job_exposes_safe_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        class Service:
            async def analyze_path(self, path: Path):
                raise RuntimeError("provider secret")

        source = tmp_path / "call.wav"
        source.write_bytes(b"audio")
        manager = AnalysisJobManager(
            analysis_service=Service(),
            max_concurrent_jobs=1,
            retention_seconds=3600,
            mapper=FakeMapper(),
        )

        accepted = manager.submit(source)
        await asyncio.sleep(0)
        failed = manager.get(accepted.job_id)
        assert failed.status == AnalysisJobStatus.FAILED
        assert failed.error == "Audio analysis failed."
        assert "provider secret" not in failed.error
        assert not source.exists()
        await manager.close()

    asyncio.run(scenario())
