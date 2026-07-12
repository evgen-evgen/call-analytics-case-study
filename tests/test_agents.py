from agents.classifier import classify
from agents.quality import assess_quality
from agents.compliance import check_compliance
from agents.summarizer import summarize


def test_classifier_detects_credit_topic():
    transcript = [{"speaker": "Оператор", "text": "Здравствуйте, хочу оформить кредит"}]
    result = classify(transcript)
    assert result["topic"] == "кредиты"
    assert result["priority"] in {"low", "medium", "high"}


def test_quality_assessment_returns_checklist():
    transcript = [
        {"speaker": "Оператор", "text": "Добрый день, спасибо за обращение"},
        {"speaker": "Клиент", "text": "Хочу уточнить по кредиту"},
    ]
    result = assess_quality(transcript)
    assert "checklist" in result
    assert result["checklist"]["greeting"] is True


def test_compliance_check_reports_safe_result():
    transcript = [{"speaker": "Оператор", "text": "Мы не обещаем доходность"}]
    result = check_compliance(transcript)
    assert result["passed"] is True


def test_summarizer_creates_action_items():
    transcript = [{"speaker": "Клиент", "text": "Нужно отправить договор по email"}]
    result = summarize(transcript)
    assert "summary" in result
    assert isinstance(result["action_items"], list)
