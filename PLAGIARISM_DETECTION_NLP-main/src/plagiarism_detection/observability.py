"""Privacy-safe request context, counters, and rate limiting."""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _valid_request_id(supplied) else str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()

    def increment(self, name: str, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            items = sorted(self._counters.items())
        lines = []
        for (name, labels), value in items:
            suffix = ""
            if labels:
                rendered = ",".join(
                    f'{key}="{_escape_label(label)}"' for key, label in labels
                )
                suffix = f"{{{rendered}}}"
            lines.append(f"sourcelens_{name}{suffix} {value}")
        return "\n".join(lines) + ("\n" if lines else "")


class SlidingWindowRateLimiter:
    def __init__(self, requests: int = 60, window_seconds: int = 60) -> None:
        if requests < 1 or window_seconds < 1:
            raise ValueError("rate-limit values must be positive")
        self.requests = requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}

    def allow(self, identity: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - self.window_seconds
        with self._lock:
            events = self._events.setdefault(identity, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests:
                return False
            events.append(timestamp)
            return True


def _valid_request_id(value: str) -> bool:
    return 0 < len(value) <= 64 and all(
        character.isalnum() or character in "-_." for character in value
    )


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
