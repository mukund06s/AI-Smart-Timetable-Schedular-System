"""
In-memory rate limiting for agent LLM repair sessions.
Prevents runaway API usage without changing core agent logic.
"""

import time
from collections import deque
from typing import Deque, Tuple

from config.settings import AGENT_SETTINGS


class AgentRateLimiter:
    """Tracks repair invocations per session key within a sliding hour window."""

    def __init__(
        self,
        max_repairs_per_hour: int = AGENT_SETTINGS.max_repairs_per_hour,
    ):
        self.max_repairs_per_hour = max_repairs_per_hour
        self._events: dict[str, Deque[float]] = {}

    def _prune(self, key: str, now: float) -> None:
        window_start = now - 3600
        events = self._events.setdefault(key, deque())
        while events and events[0] < window_start:
            events.popleft()

    def allow(self, session_key: str = "default") -> Tuple[bool, str]:
        now = time.time()
        self._prune(session_key, now)
        events = self._events.setdefault(session_key, deque())
        if len(events) >= self.max_repairs_per_hour:
            return (
                False,
                f"Agent repair rate limit reached ({self.max_repairs_per_hour}/hour). "
                "Try again later or contact an administrator.",
            )
        events.append(now)
        return True, ""

    def remaining(self, session_key: str = "default") -> int:
        now = time.time()
        self._prune(session_key, now)
        used = len(self._events.get(session_key, ()))
        return max(self.max_repairs_per_hour - used, 0)


_GLOBAL_LIMITER = AgentRateLimiter()


def get_rate_limiter() -> AgentRateLimiter:
    return _GLOBAL_LIMITER
