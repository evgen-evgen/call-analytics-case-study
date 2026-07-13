from typing import Any, Dict, List


def classify(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = " ".join(item.get("text", "") for item in transcript).lower()

    if "кредит" in text:
        topic = "кредиты"
    elif "карт" in text:
        topic = "карты"
    elif "перев" in text:
        topic = "переводы"
    elif "жалоб" in text or "проблем" in text:
        topic = "жалобы"
    else:
        topic = "другое"

    if topic in {"кредиты", "жалобы"}:
        priority = "high"
    elif topic == "карты":
        priority = "medium"
    else:
        priority = "low"

    return {"topic": topic, "priority": priority}
