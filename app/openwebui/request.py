from dataclasses import dataclass
from typing import Any

from app.asr import find_audio_source
from app.schemas import AudioSource


@dataclass(frozen=True)
class OpenWebUIRequest:
    user_message: str
    model_id: str
    messages: list[dict[str, Any]]
    body: dict[str, Any]
    audio_source: AudioSource | None
    internal_task: str | None

    @property
    def stream(self) -> bool:
        return bool(self.body.get("stream"))


class OpenWebUIRequestParser:
    def parse(
        self,
        *,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> OpenWebUIRequest:
        internal_task = self._internal_task(body)
        return OpenWebUIRequest(
            user_message=user_message,
            model_id=model_id,
            messages=messages,
            body=body,
            audio_source=(
                None
                if internal_task is not None
                else find_audio_source(user_message)
            ),
            internal_task=internal_task,
        )

    @staticmethod
    def _internal_task(body: dict[str, Any]) -> str | None:
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            return None
        task = metadata.get("task")
        return str(task) if task else None
