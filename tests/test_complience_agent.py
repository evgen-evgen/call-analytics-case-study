import pytest

from app.agents.compliance import ComplianceAgent
from app.schemas import (
    ComplianceAssessment,
    ComplianceStatus,
    DisclaimerCheck,
    RecommendationCorrectness,
    TranscriptSegment,
)


class FakeLLMClient:
    def __init__(
        self,
        result: ComplianceAssessment,
    ) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model,
        temperature: float,
        response_format_mode: str,
    ) -> ComplianceAssessment:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
                "temperature": temperature,
                "response_format_mode": response_format_mode,
            }
        )

        return self.result


def make_result() -> ComplianceAssessment:
    return ComplianceAssessment(
        prohibited_phrases_found=False,
        disclaimer=DisclaimerCheck(
            required=False,
            present=None,
            reason="Обязательное предупреждение не требуется.",
            evidence=None,
        ),
        recommendation_correctness=(
            RecommendationCorrectness(
                status=ComplianceStatus.COMPLIANT,
                reason="Рекомендация безопасна.",
                evidence=None,
            )
        ),
        issues=[],
    )


@pytest.mark.asyncio
async def test_compliance_agent_calls_llm_with_schema() -> None:
    llm_client = FakeLLMClient(
        make_result()
    )

    agent = ComplianceAgent(
        llm_client
    )

    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text=(
                "Не сообщайте PIN, CVV "
                "и коды из SMS."
            ),
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=3.2,
            end=5.0,
            text="Хорошо.",
        ),
    ]

    result = await agent.run(
        transcript
    )

    assert result == make_result()
    assert len(llm_client.calls) == 1

    call = llm_client.calls[0]

    assert (
        call["response_model"]
        is ComplianceAssessment
    )
    assert call["temperature"] == 0.0
    assert call["response_format_mode"] == "json_object"
    assert "PIN" in call["system_prompt"]
    assert "not_verifiable" in call["system_prompt"]
    assert "комплаенс-контроля" in call["system_prompt"]
    assert "обязательные предупреждения" in call["system_prompt"]
    assert "комплаенс-риски" in call["user_prompt"]
    assert "Оператор" in call["user_prompt"]
    assert "только на русском языке" in call["system_prompt"]


@pytest.mark.asyncio
async def test_compliance_agent_preserves_transcript() -> None:
    llm_client = FakeLLMClient(
        make_result()
    )

    agent = ComplianceAgent(
        llm_client
    )

    transcript = [
        TranscriptSegment(
            speaker="Оператор",
            start=0.0,
            end=3.0,
            text=(
                "Я гарантирую, что кредит "
                "точно одобрят."
            ),
        ),
    ]

    await agent.run(
        transcript
    )

    prompt = llm_client.calls[0][
        "user_prompt"
    ]

    assert (
        "Я гарантирую, что кредит точно одобрят."
        in prompt
    )
