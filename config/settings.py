"""
Centralized tunable settings (env-overridable).
Does not change runtime behavior unless env vars are set.
"""

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class AgentSettings:
    max_turns: int = _int_env("AGENT_MAX_TURNS", 10)
    max_repairs_per_hour: int = _int_env("AGENT_MAX_REPAIRS_PER_HOUR", 30)
    llm_max_retries: int = _int_env("AGENT_LLM_MAX_RETRIES", 3)
    llm_retry_delay_seconds: float = float(os.getenv("AGENT_LLM_RETRY_DELAY", "0.5"))


@dataclass(frozen=True)
class AppSettings:
    ga_max_attempts: int = _int_env("GA_MAX_ATTEMPTS", 2000)
    ga_repair_rounds: int = _int_env("GA_REPAIR_ROUNDS", 25)
    health_check_timeout_seconds: int = _int_env("HEALTH_CHECK_TIMEOUT", 10)


AGENT_SETTINGS = AgentSettings()
APP_SETTINGS = AppSettings()
