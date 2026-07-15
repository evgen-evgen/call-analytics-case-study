import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.agents.classifier import ClassificationAgent
from app.agents.compliance import ComplianceAgent
from app.agents.quality import QualityAgent
from app.agents.summarizer import SummarizerAgent
from app.llm.client import build_strict_json_schema
from app.schemas import (
    ClassificationResult,
    ComplianceAssessment,
    QualityAssessment,
    QualityResult,
    SummaryAssessment,
    TranscriptSegment,
)


AGENT_SCHEMA_CASES = (
    pytest.param(
        ClassificationAgent,
        ClassificationResult,
        {
            "topic": "карты",
            "priority": "high",
            "reasoning": "Клиент сообщил о неизвестной операции.",
        },
        {
            "topic": "страхование",
            "priority": "urgent",
            "reasoning": "",
        },
        id="classifier",
    ),
    pytest.param(
        QualityAgent,
        QualityAssessment,
        {
            "checklist": {
                "greeting": {"passed": True, "reason": "Есть приветствие."},
                "need_identification": {"passed": True, "reason": "Запрос уточнён."},
                "solution": {"passed": True, "reason": "Решение дано."},
                "farewell": {"passed": False, "reason": "Нет прощания."},
            },
        },
        {
            "checklist": {
                "greeting": {"passed": True, "reason": "Есть приветствие."},
            },
        },
        id="quality",
    ),
    pytest.param(
        ComplianceAgent,
        ComplianceAssessment,
        {
            "prohibited_phrases_found": True,
            "disclaimer": {
                "required": False,
                "present": None,
                "reason": "Не требуется.",
            },
            "recommendation_correctness": {
                "status": "violation",
                "reason": "Запрошены секретные данные.",
            },
            "issues": [
                {
                    "category": "unsafe_data_request",
                    "severity": "critical",
                    "description": "Оператор запросил PIN-код.",
                    "recommendation": "Не запрашивать PIN-код.",
                }
            ],
        },
        {
            "prohibited_phrases_found": False,
            "issues": [],
        },
        id="compliance",
    ),
    pytest.param(
        SummarizerAgent,
        SummaryAssessment,
        {
            "summary": (
                "Клиент сообщил о неизвестной операции. "
                "Оператор принял запрос. "
                "Карту нужно заблокировать."
            ),
            "action_items": [
                {
                    "action": "Заблокировать карту.",
                    "owner": "Клиент",
                    "reason": "Исключить новые операции.",
                }
            ],
        },
        {
            "summary": "Итог разговора",
            "action_items": "Заблокировать карту",
        },
        id="summarizer",
    ),
)


class SchemaValidatingLLMClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.received_model: type[BaseModel] | None = None

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float,
        response_format_mode: str = "json_schema",
    ) -> BaseModel:
        assert system_prompt
        assert "тестовый разговор" in user_prompt.lower()
        assert temperature == 0.0
        assert response_format_mode in {"json_schema", "json_object"}
        self.received_model = response_model
        return response_model.model_validate(self.payload)


@pytest.mark.parametrize(
    ("agent_type", "result_model", "valid_payload", "invalid_payload"),
    AGENT_SCHEMA_CASES,
)
def test_agent_uses_its_schema_and_accepts_valid_result(
    agent_type: type,
    result_model: type[BaseModel],
    valid_payload: dict[str, Any],
    invalid_payload: dict[str, Any],
) -> None:
    client = SchemaValidatingLLMClient(valid_payload)
    agent = agent_type(client)
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0.0,
            end=2.0,
            text="Тестовый разговор.",
        )
    ]

    result = asyncio.run(agent.run(transcript))

    assert client.received_model is result_model
    assert isinstance(result, result_model)


@pytest.mark.parametrize(
    ("agent_type", "result_model", "valid_payload", "invalid_payload"),
    AGENT_SCHEMA_CASES,
)
def test_agent_schema_rejects_invalid_result(
    agent_type: type,
    result_model: type[BaseModel],
    valid_payload: dict[str, Any],
    invalid_payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        result_model.model_validate(invalid_payload)


@pytest.mark.parametrize(
    ("agent_type", "result_model", "valid_payload", "invalid_payload"),
    AGENT_SCHEMA_CASES,
)
def test_agent_schema_is_strict_for_llm_provider(
    agent_type: type,
    result_model: type[BaseModel],
    valid_payload: dict[str, Any],
    invalid_payload: dict[str, Any],
) -> None:
    schema = build_strict_json_schema(result_model)
    _assert_all_objects_are_strict(schema)


def _assert_all_objects_are_strict(node: object) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_all_objects_are_strict(item)
        return

    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    if isinstance(properties, dict):
        assert node.get("additionalProperties") is False
        assert set(node.get("required", [])) == set(properties)

    for value in node.values():
        _assert_all_objects_are_strict(value)
