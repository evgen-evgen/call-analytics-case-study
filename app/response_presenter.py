class ResponsePresenter:
    _PROGRESS_MESSAGES = {
        "file_received": "📎 Файл получен. Начинаю обработку…",
        "audio_prepared": "🎧 Аудио подготовлено. Выполняю транскрибацию…",
        "transcription_completed": "📝 Транскрибация готова. Разделяю спикеров…",
        "diarization_completed": "👥 Спикеры определены. Готовлю финальный транскрипт…",
        "analysis_started": "🧠 Запускаю аналитических агентов…",
        "analysis_completed": "✅ Анализ готов.",
    }

    def format_progress(self, event: str) -> str:
        return self._PROGRESS_MESSAGES[event] + "\n\n"
