from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ActionItemOwner(StrEnum):
    OPERATOR = "Оператор"
    CLIENT = "Клиент"
    BANK = "Банк"
    UNKNOWN = "Не определено"


class ActionItemStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"


class ActionItem(BaseModel):
    action: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Конкретное действие, которое должно быть выполнено "
            "после звонка или уже было инициировано, но не завершено."
        ),
    )

    owner: ActionItemOwner

    status: ActionItemStatus = ActionItemStatus.PENDING

    deadline: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Срок, указанный в разговоре. "
            "Null, если срок не был назван."
        ),
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Краткое объяснение, почему действие требуется."
        ),
    )


class SummaryAssessment(BaseModel):
    summary: str = Field(
        min_length=20,
        max_length=2000,
        description=(
            "Краткое резюме разговора из 3–5 предложений."
        ),
    )

    action_items: list[ActionItem] = Field(
        default_factory=list,
        max_length=10,
    )

    @model_validator(mode="after")
    def validate_summary_sentences(
        self,
    ) -> "SummaryAssessment":
        sentence_count = _count_sentences(
            self.summary
        )

        if not 3 <= sentence_count <= 5:
            raise ValueError(
                "Summary must contain from 3 to 5 sentences."
            )

        return self


def _count_sentences(value: str) -> int:
    import re

    sentences = re.findall(
        r"[^.!?]+[.!?]+|[^.!?]+$",
        value.strip(),
    )

    return len(
        [
            sentence
            for sentence in sentences
            if sentence.strip()
        ]
    )