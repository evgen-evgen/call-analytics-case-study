import json
from typing import Any


class ResponsePresenter:
    _PROGRESS_MESSAGES = {
        "audio_queued": (
            "⏳ Другой аудиофайл уже обрабатывается. "
            "Ваш запрос поставлен в очередь…"
        ),
        "file_received": "📎 Файл получен. Начинаю обработку…",
        "audio_prepared": "🎧 Аудио подготовлено. Выполняю транскрибацию…",
        "transcription_completed": "📝 Транскрибация готова. Разделяю спикеров…",
        "diarization_completed": "👥 Спикеры определены. Готовлю финальный транскрипт…",
        "analysis_started": "🧠 Запускаю аналитических агентов…",
        "analysis_completed": "✅ Анализ готов.",
    }

    def format_progress(self, event: str) -> str:
        return self._PROGRESS_MESSAGES[event] + "\n\n"

    def format_analysis(self, analysis: Any, marker: str) -> str:
        transcript_json = json.dumps(
            [segment.model_dump() for segment in analysis.transcript],
            ensure_ascii=False,
            indent=2,
        )
        quality = analysis.quality
        checklist = quality.checklist
        compliance = analysis.compliance
        summary = analysis.summary

        quality_issues = self._format_list(
            quality.issues,
            empty_text="Замечаний нет.",
        )
        compliance_issues = self._format_compliance_issues(
            compliance.issues
        )
        action_items = self._format_list(
            summary.action_items,
            empty_text="Дополнительных действий нет.",
        )

        return (
            f"{marker}\n"
            "## Транскрипция\n\n"
            "```json\n"
            f"{transcript_json}\n"
            "```\n\n"
            "## Классификация\n\n"
            f"- **Тема:** {analysis.classification.topic.value}\n"
            f"- **Приоритет:** {analysis.classification.priority.value}\n"
            f"- **Обоснование:** {analysis.classification.reasoning}\n\n"
            "## Качество обслуживания\n\n"
            f"- **Оценка:** {quality.total}/100\n"
            f"- **Приветствие:** {self._yes_no(checklist.greeting)}\n"
            f"- **Выявление потребности:** {self._yes_no(checklist.need_detection)}\n"
            f"- **Решение предложено:** {self._yes_no(checklist.solution_provided)}\n"
            f"- **Прощание:** {self._yes_no(checklist.farewell)}\n\n"
            "### Замечания\n\n"
            f"{quality_issues}\n\n"
            "## Compliance\n\n"
            f"**Статус:** {'✅ Проверка пройдена' if compliance.passed else '❌ Обнаружены нарушения'}\n\n"
            f"{compliance_issues}\n\n"
            "## Резюме\n\n"
            f"{summary.summary}\n\n"
            "### Следующие действия\n\n"
            f"{action_items}"
        )

    @staticmethod
    def _yes_no(value: bool) -> str:
        return "✅ Да" if value else "❌ Нет"

    @staticmethod
    def _format_list(items: list[str], *, empty_text: str) -> str:
        if not items:
            return empty_text
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _format_compliance_issues(issues: list[Any]) -> str:
        if not issues:
            return "Нарушений не обнаружено."

        blocks: list[str] = []
        for issue in issues:
            details = [
                f"- **{issue.category}:** {issue.description}"
            ]
            if issue.quote:
                details.append(f"  - Цитата: «{issue.quote}»")
            if issue.start is not None:
                timecode = f"{issue.start:.1f} с"
                if issue.end is not None:
                    timecode += f" — {issue.end:.1f} с"
                details.append(f"  - Таймкод: {timecode}")
            blocks.append("\n".join(details))
        return "\n".join(blocks)
