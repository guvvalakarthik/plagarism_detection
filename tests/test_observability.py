from plagiarism_detection.observability import (
    MetricsRegistry,
    SlidingWindowRateLimiter,
)


def test_rate_limiter_expires_old_requests() -> None:
    limiter = SlidingWindowRateLimiter(requests=2, window_seconds=10)

    assert limiter.allow("workspace", now=0)
    assert limiter.allow("workspace", now=1)
    assert not limiter.allow("workspace", now=2)
    assert limiter.allow("workspace", now=11)


def test_metrics_render_without_sensitive_values() -> None:
    metrics = MetricsRegistry()
    metrics.increment("jobs_total", status="ready")
    metrics.increment("jobs_total", status="ready")

    rendered = metrics.render_prometheus()

    assert 'sourcelens_jobs_total{status="ready"} 2' in rendered
