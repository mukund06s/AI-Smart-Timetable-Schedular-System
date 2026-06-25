"""
Central logging configuration for the timetable scheduling system.
Supports optional Sentry error tracking via SENTRY_DSN environment variable.
"""

import logging
import os
import sys
from typing import Optional


_CONFIGURED = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logger once with a consistent format."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    _init_sentry_if_configured()
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name)


def _init_sentry_if_configured() -> None:
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=os.getenv("APP_ENV", "development"),
        )
        logging.getLogger(__name__).info("Sentry error tracking enabled")
    except ImportError:
        logging.getLogger(__name__).warning(
            "SENTRY_DSN set but sentry-sdk not installed; skipping Sentry init"
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("Sentry initialization failed: %s", exc)


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """Log exception with stack trace; forwards to Sentry when configured."""
    logger.exception("%s: %s", message, exc)
    try:
        import sentry_sdk

        sentry_sdk.capture_exception(exc)
    except ImportError:
        pass
