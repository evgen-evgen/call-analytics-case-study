from types import SimpleNamespace

from app.mappers import AnalyzeResponseMapper
from app.api.main import app
from app.schemas import TranscriptSegment


def test_present_analysis_matches_public_api_contract() -> None:
    passed = SimpleNamespace(passed=True)
    result = SimpleNamespace(
        transcript=[
            TranscriptSegment(
                speaker="Оператор",
                start=0,
                end=1,
                text="Добрый день.",
            )
        ],
        classification=SimpleNamespace(
            topic=SimpleNamespace(value="кредиты"),
            priority=SimpleNamespace(value="medium"),
        ),
        quality=SimpleNamespace(
            score=75,
            checklist=SimpleNamespace(
                greeting=passed,
                need_identification=passed,
                solution=passed,
                farewell=SimpleNamespace(passed=False),
            ),
        ),
        compliance=SimpleNamespace(
            passed=True,
            assessment=SimpleNamespace(issues=[]),
        ),
        summary=SimpleNamespace(
            summary="Клиент обратился по кредиту.",
            action_items=[SimpleNamespace(action="Отправить условия")],
        ),
    )

    response = AnalyzeResponseMapper().map(result).model_dump()

    assert response["quality_score"] == {
        "total": 75,
        "checklist": {
            "greeting": True,
            "need_detection": True,
            "solution_provided": True,
            "farewell": False,
        },
    }
    assert response["classification"] == {
        "topic": "кредиты",
        "priority": "medium",
    }
    assert response["action_items"] == ["Отправить условия"]


def test_analysis_api_uses_async_job_contract() -> None:
    schema = app.openapi()

    submit = schema["paths"]["/analyze"]["post"]
    assert "202" in submit["responses"]
    assert "/analyses/{job_id}" in schema["paths"]
