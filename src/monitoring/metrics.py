"""
Prometheus metrics for production monitoring.

WHY PROMETHEUS?
---------------
Once a model is deployed, you need to know:
- Is it responding? (uptime)
- Is it fast enough? (latency)
- Is it being used? (request rate)
- Is it failing? (error rate)

Prometheus scrapes these metrics every 15 seconds.
Grafana reads from Prometheus to build dashboards.
Alerts fire when metrics cross thresholds.

This is the observability layer that separates production systems
from demo projects.
"""

import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# ── Counters (monotonically increasing) ──────────────────────────────────────

REQUEST_COUNT = Counter(
    "llmops_requests_total",
    "Total number of inference requests",
    ["endpoint", "status"]
)

ERROR_COUNT = Counter(
    "llmops_errors_total",
    "Total number of errors",
    ["error_type"]
)

# ── Histograms (track distributions) ─────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "llmops_request_latency_ms",
    "Request latency in milliseconds",
    ["endpoint"],
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

TOKEN_COUNT = Histogram(
    "llmops_tokens_generated",
    "Number of tokens generated per request",
    buckets=[10, 25, 50, 100, 150, 200, 256]
)

# ── Gauges (can go up and down) ───────────────────────────────────────────────

ACTIVE_REQUESTS = Gauge(
    "llmops_active_requests",
    "Number of requests currently being processed"
)

MODEL_LOADED = Gauge(
    "llmops_model_loaded",
    "Whether the model is loaded and ready (1=yes, 0=no)"
)

LAST_REQUEST_TIMESTAMP = Gauge(
    "llmops_last_request_timestamp",
    "Unix timestamp of last inference request"
)

# ── Info (static labels) ──────────────────────────────────────────────────────

MODEL_INFO = Gauge(
    "llmops_model_info",
    "Model metadata",
    ["model_id", "model_type"]
)


# ── Helper functions ──────────────────────────────────────────────────────────

def record_request(
    endpoint: str,
    latency_ms: float,
    success: bool = True,
    tokens: int = None
) -> None:
    """
    Records a completed inference request.
    Call this after every /generate request.
    """
    status = "success" if success else "error"
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_ms)
    LAST_REQUEST_TIMESTAMP.set(time.time())

    if tokens is not None:
        TOKEN_COUNT.observe(tokens)


def set_model_loaded(loaded: bool) -> None:
    """Updates model loaded gauge."""
    MODEL_LOADED.set(1 if loaded else 0)


def get_metrics_text() -> str:
    """Returns Prometheus metrics as text."""
    return generate_latest().decode("utf-8")


if __name__ == "__main__":
    print("Testing Prometheus metrics...\n")

    # Simulate some requests
    print("Simulating 10 requests...")
    for i in range(10):
        latency = 200 + (i * 50)  # 200ms to 650ms
        tokens = 50 + (i * 10)
        success = i < 9  # last one fails

        record_request(
            endpoint="/generate",
            latency_ms=latency,
            success=success,
            tokens=tokens
        )

    # Simulate 2 errors
    ERROR_COUNT.labels(error_type="model_not_loaded").inc()
    ERROR_COUNT.labels(error_type="inference_error").inc()

    # Set model as loaded
    set_model_loaded(True)
    MODEL_INFO.labels(
        model_id="djism/phi3-medical-qa",
        model_type="fine-tuned-lora"
    ).set(1)

    print("\nMetrics output (Prometheus format):")
    print("=" * 55)
    metrics_text = get_metrics_text()
    # Print only our custom metrics
    for line in metrics_text.split("\n"):
        if "llmops" in line:
            print(line)

    print("\n✅ Prometheus metrics working correctly!")
    print(f"\nTo view in Grafana:")
    print(f"  1. Start API: python src/serving/api.py")
    print(f"  2. Visit: http://localhost:8000/metrics")
    print(f"  3. Prometheus scrapes this endpoint every 15s")