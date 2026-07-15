import os

import pytest

from app.agents.compliance import ComplianceAgent
from app.services.compliance_service import (
    build_compliance_result,
    validate_compliance_evidence,
)
from app.llm.client import LLMClient
from app.schemas import (
    ComplianceSeverity,
    ComplianceStatus,
    TranscriptSegment,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("LLM_API_KEY"),
        reason="LLM_API_KEY is not configured",
    ),
]


@pytest.mark.asyncio
async def test_detects_pin_request_as_critical() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text=(
                "Назовите PIN-код вашей карты, "
                "чтобы я проверил операцию."
            ),
        ),
    ]

    client = LLMClient()
    agent = ComplianceAgent(client)

    assessment = await agent.run(
        transcript
    )

    assessment = validate_compliance_evidence(
        assessment,
        transcript,
    )

    result = build_compliance_result(
        assessment
    )

    assert result.passed is False
    assert (
        result.risk_level
        == ComplianceSeverity.CRITICAL
    )
    assert result.issue_count >= 1


@pytest.mark.asyncio
async def test_credit_guarantee_requires_violation() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=4.0,
            text=(
                "Я гарантирую, что кредит "
                "вам точно одобрят."
            ),
        ),
    ]

    client = LLMClient()
    agent = ComplianceAgent(client)

    assessment = await agent.run(
        transcript
    )

    result = build_compliance_result(
        assessment
    )

    assert result.passed is False
    assert result.issue_count >= 1
    assert assessment.disclaimer.required is True


@pytest.mark.asyncio
async def test_unknown_interest_rate_is_not_verifiable() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=4.0,
            text=(
                "Ставка по кредиту составляет "
                "двенадцать процентов."
            ),
        ),
    ]

    client = LLMClient()
    agent = ComplianceAgent(client)

    assessment = await agent.run(
        transcript
    )

    assert (
        assessment.recommendation_correctness.status
        in {
            ComplianceStatus.NOT_VERIFIABLE,
            ComplianceStatus.COMPLIANT,
        }
    )

    # Само отсутствие базы тарифов не должно
    # автоматически создавать нарушение.
    assert not any(
        issue.description.lower().startswith(
            "не удалось проверить ставку"
        )
        for issue in assessment.issues
    )


@pytest.mark.asyncio
async def test_safe_warning_passes() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=4.0,
            text=(
                "Не сообщайте PIN, CVV "
                "и коды подтверждения из SMS."
            ),
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=4.2,
            end=8.0,
            text=(
                "Я заблокирую карту и зарегистрирую "
                "обращение."
            ),
        ),
    ]

    client = LLMClient()
    agent = ComplianceAgent(client)

    assessment = await agent.run(
        transcript
    )

    result = build_compliance_result(
        assessment
    )

    assert result.passed is True
    assert result.issue_count == 0
