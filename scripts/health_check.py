#!/usr/bin/env python3
"""
Health check script for deployment and CI smoke tests.
Exit code 0 = healthy, 1 = unhealthy.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_imports() -> bool:
    required = [
        "streamlit",
        "pandas",
        "firebase_admin",
        "anthropic",
        "networkx",
        "scipy",
    ]
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print(f"FAIL imports: missing {', '.join(missing)}")
        return False
    print("OK imports")
    return True


def check_project_modules() -> bool:
    try:
        from agent.timetable_agent import TimetableAgent
        from agent.integration import run_agentic_clash_repair
        from utils.logging_config import configure_logging
        from agent.rate_limiter import get_rate_limiter
        from agent.input_validation import validate_tool_args

        configure_logging()
        assert validate_tool_args("tool_move_class", {}) is not None
        allowed, _ = get_rate_limiter().allow("health-check")
        assert allowed
        print("OK project modules")
        return True
    except Exception as exc:
        print(f"FAIL project modules: {exc}")
        return False


def check_secrets_configured() -> bool:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    has_env = bool(os.getenv("ANTHROPIC_API_KEY"))
    if secrets_path.exists() or has_env:
        print("OK secrets (file or ANTHROPIC_API_KEY env present)")
        return True
    print("WARN secrets not configured (expected for CI; set before production)")
    return True


def check_firebase_optional() -> bool:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        print("SKIP firebase (no secrets.toml)")
        return True
    try:
        import streamlit as st
        from firebase_admin import credentials, firestore
        import firebase_admin

        if not firebase_admin._apps:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        list(db.collection("logs").limit(1).stream())
        print("OK firebase connectivity")
        return True
    except Exception as exc:
        print(f"WARN firebase check skipped/failed: {exc}")
        return True


def main() -> int:
    checks = [
        check_imports(),
        check_project_modules(),
        check_secrets_configured(),
        check_firebase_optional(),
    ]
    if all(checks):
        print("Health check passed")
        return 0
    print("Health check failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
