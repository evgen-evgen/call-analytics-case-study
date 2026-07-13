from typing import Any, Dict, List


FORBIDDEN_PHRASES = ["гарантируем доходность", "обещаем прибыль"]


def check_compliance(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = " ".join(item.get("text", "") for item in transcript).lower()
    issues = [phrase for phrase in FORBIDDEN_PHRASES if phrase in text]
    return {"passed": not issues, "issues": issues}
