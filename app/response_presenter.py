import json
from typing import Any

from app.schemas import AnalyzeResponse


class ResponsePresenter:
    _COMPLIANCE_CATEGORIES = {
        "prohibited_phrase": "запрещённая фраза",
        "missing_disclaimer": "отсутствует обязательное предупреждение",
        "incorrect_recommendation": "некорректная рекомендация",
        "unsafe_data_request": "запрос секретных данных",
        "misleading_promise": "вводящее в заблуждение обещание",
        "other": "другое нарушение",
    }
    _COMPLIANCE_SEVERITIES = {
        "low": "низкий риск",
        "medium": "средний риск",
        "high": "высокий риск",
        "critical": "критический риск",
    }
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

    def format_analysis(
        self,
        analysis: AnalyzeResponse,
        marker: str,
    ) -> str:
        transcript_json = json.dumps(
            [segment.model_dump() for segment in analysis.transcript],
            ensure_ascii=False,
            indent=2,
        )
        failed_sections = {
            "classification": "Классификация недоступна.",
            "quality": "Оценка качества недоступна.",
            "compliance": "Комплаенс-проверка недоступна.",
            "summary": "Резюме недоступно.",
        }

        if analysis.agent_errors:
            failed = ", ".join(sorted(analysis.agent_errors))
            failure_notice = (
                "## Неполный анализ\n\n"
                f"⚠️ Не удалось получить результат агентов: {failed}. "
                "Результаты остальных агентов сохранены.\n\n"
            )
        else:
            failure_notice = ""

        classification_text = failed_sections["classification"]
        if analysis.classification is not None:
            classification_text = (
                f"- **Тема:** {analysis.classification.topic}\n"
                f"- **Приоритет:** {analysis.classification.priority}"
            )

        quality = analysis.quality_score
        if quality is not None:
            checklist = quality.checklist
            quality_text = (
                f"- **Оценка:** {quality.total}/100\n"
                f"- **Приветствие:** {self._yes_no(checklist.greeting)}\n"
                f"- **Выявление потребности:** {self._yes_no(checklist.need_detection)}\n"
                f"- **Решение предложено:** {self._yes_no(checklist.solution_provided)}\n"
                f"- **Прощание:** {self._yes_no(checklist.farewell)}"
            )
            quality_issues = self._format_list(
                [
                    label
                    for label, passed in (
                        ("Нет приветствия.", checklist.greeting),
                        ("Потребность клиента не выявлена.", checklist.need_detection),
                        ("Решение не предложено.", checklist.solution_provided),
                        ("Нет прощания.", checklist.farewell),
                    )
                    if not passed
                ],
                empty_text="Замечаний нет.",
            )
        else:
            quality_text = failed_sections["quality"]
            quality_issues = ""

        compliance = analysis.compliance
        if compliance is not None:
            compliance_status = (
                "✅ Проверка пройдена"
                if compliance.passed
                else "❌ Обнаружены нарушения"
            )
            compliance_issues = self._format_compliance_issues(compliance.issues)
        else:
            compliance_status = failed_sections["compliance"]
            compliance_issues = ""
        action_items = self._format_list(
            analysis.action_items,
            empty_text="Дополнительных действий нет.",
        )

        return (
            f"{marker}\n"
            f"{failure_notice}"
            "## Транскрипция\n\n"
            "```json\n"
            f"{transcript_json}\n"
            "```\n\n"
            "## Классификация\n\n"
            f"{classification_text}\n\n"
            "## Качество обслуживания\n\n"
            f"{quality_text}\n\n"
            "### Замечания\n\n"
            f"{quality_issues}\n\n"
            "## Комплаенс-проверка\n\n"
            f"**Статус:** {compliance_status}\n\n"
            f"{compliance_issues}\n\n"
            "## Резюме\n\n"
            f"{analysis.summary or failed_sections['summary']}\n\n"
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
            category = ResponsePresenter._COMPLIANCE_CATEGORIES.get(
                str(issue.category),
                "нарушение",
            )
            severity = ResponsePresenter._COMPLIANCE_SEVERITIES.get(
                str(issue.severity),
                "риск не определён",
            )
            details = [
                f"- **{category} ({severity}):** "
                f"{issue.description}",
                f"  - Рекомендация: {issue.recommendation}",
            ]
            evidence = issue.evidence
            if evidence is not None:
                details.append(f"  - Цитата: «{evidence.quote}»")
                timecode = f"{evidence.start:.1f} с"
                if evidence.end is not None:
                    timecode += f" — {evidence.end:.1f} с"
                details.append(f"  - Таймкод: {timecode}")
            blocks.append("\n".join(details))
        return "\n".join(blocks)
