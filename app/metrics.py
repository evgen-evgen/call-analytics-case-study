from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    start_http_server,
)

from app.schemas.classification import CallTopic

if TYPE_CHECKING:
    from app.agents.supervisor import AnalysisResult


class CallAnalyticsMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else REGISTRY
        self.calls = Counter(
            "call_analytics_calls_total",
            "Number of successfully analyzed calls.",
            labelnames=("topic",),
            registry=self.registry,
        )
        self.quality_score = Histogram(
            "call_analytics_quality_score",
            "Quality score of analyzed calls from 0 to 100.",
            buckets=(0, 25, 50, 75, 90, 100),
            registry=self.registry,
        )
        for topic in CallTopic:
            self.calls.labels(topic=topic.value).inc(0)

    def record_analysis(self, analysis: AnalysisResult) -> None:
        if analysis.classification is not None:
            self.calls.labels(topic=analysis.classification.topic.value).inc()
        if analysis.quality is not None:
            self.quality_score.observe(analysis.quality.score)


metrics = CallAnalyticsMetrics()
_server_lock = threading.Lock()
_server_started = False


def start_metrics_server(port: int = 9100) -> None:
    global _server_started
    with _server_lock:
        if _server_started:
            return
        start_http_server(port)
        _server_started = True
