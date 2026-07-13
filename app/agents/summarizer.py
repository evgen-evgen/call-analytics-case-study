from typing import Any, Dict, List


def summarize(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = " ".join(item.get("text", "") for item in transcript)
    summary = f"Клиент обратился с запросом по теме: {text[:80]}"
    action_items = []
    if "кредит" in text.lower():
        action_items.append("Подготовить информацию по кредитному продукту")
    if "email" in text.lower():
        action_items.append("Отправить документы клиенту")
    return {"summary": summary, "action_items": action_items}
