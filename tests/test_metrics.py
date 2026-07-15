from types import SimpleNamespace

from prometheus_client import CollectorRegistry

from app.metrics import CallAnalyticsMetrics


def test_records_call_topic_and_quality_score() -> None:
    registry = CollectorRegistry()
    metrics = CallAnalyticsMetrics(registry)
    analysis = SimpleNamespace(
        classification=SimpleNamespace(
            topic=SimpleNamespace(value="карты")
        ),
        quality=SimpleNamespace(score=75),
    )

    metrics.record_analysis(analysis)

    assert registry.get_sample_value(
        "call_analytics_calls_total",
        {"topic": "карты"},
    ) == 1
    assert registry.get_sample_value(
        "call_analytics_quality_score_sum"
    ) == 75
    assert registry.get_sample_value(
        "call_analytics_quality_score_count"
    ) == 1
