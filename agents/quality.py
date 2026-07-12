from typing import Any, Dict, List


def assess_quality(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = " ".join(item.get("text", "") for item in transcript).lower()
    greeting = "добрый" in text or "здравствуйте" in text
    need_detection = "хочу" in text or "нужно" in text or "нуж" in text
    solution_provided = "можно" in text or "помож" in text or "отправ" in text
    farewell = "до свидания" in text or "спасибо" in text

    checklist = {
        "greeting": greeting,
        "need_detection": need_detection,
        "solution_provided": solution_provided,
        "farewell": farewell,
    }

    total = sum(25 for value in checklist.values() if value)
    return {"total": total, "checklist": checklist}
