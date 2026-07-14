import json
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel


from app.llm.client import LLMClient
from app.schemas import TranscriptSegment


AgentResultT = TypeVar(
    "AgentResultT",
    bound=BaseModel,
)


class BaseAgent(
    ABC,
    Generic[AgentResultT],
):
    result_model: type[AgentResultT]
    system_prompt: str

    def __init__(
        self,
        llm_client: LLMClient,
    ) -> None:
        self.llm_client = llm_client

    async def run(
        self,
        transcript: list[TranscriptSegment],
    ) -> AgentResultT:
        return await self.llm_client.generate_structured(
            system_prompt=self.system_prompt,
            user_prompt=self.build_user_prompt(
                transcript
            ),
            response_model=self.result_model,
            temperature=0.0,
        )

    @abstractmethod
    def build_user_prompt(
        self,
        transcript: list[TranscriptSegment],
    ) -> str:
        raise NotImplementedError

    @staticmethod
    def format_transcript(
        transcript: list[TranscriptSegment],
    ) -> str:
        transcript_data = [
            segment.model_dump()
            for segment in transcript
        ]

        return json.dumps(
            transcript_data,
            ensure_ascii=False,
            indent=2,
        )
