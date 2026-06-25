"""
Firebase operations for the Agentic AI layer.
Uses FirebaseManager agent methods when available.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


DEFAULT_AGENT_CONFIG = {
    "max_turns": 10,
    "llm_model": "claude-sonnet-4-5",
    "enabled": True,
    "fallback_to_random_repair": True,
}


class AgentFirebaseOps:
    """Wrapper for agent-specific Firebase read/write operations."""

    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager

    def get_agent_config(self, config_id: str = "default") -> dict:
        if self.firebase and hasattr(self.firebase, "get_agent_config"):
            return self.firebase.get_agent_config(config_id)
        return dict(DEFAULT_AGENT_CONFIG)

    def save_agent_session(self, session_id: str, session_data: dict) -> tuple:
        if self.firebase and hasattr(self.firebase, "save_agent_session"):
            return self.firebase.save_agent_session(session_id, session_data)

        if self.firebase and getattr(self.firebase, "db", None):
            try:
                from google.cloud import firestore

                session_data["updated_at"] = firestore.SERVER_TIMESTAMP
                self.firebase.db.collection("agent_sessions").document(session_id).set(
                    session_data
                )
                return True, session_id
            except Exception as exc:
                return False, str(exc)

        return False, "Firebase not available"

    def save_repair_history(self, repair_id: str, repair_data: dict) -> tuple:
        if self.firebase and hasattr(self.firebase, "save_repair_history"):
            return self.firebase.save_repair_history(repair_id, repair_data)

        if self.firebase and getattr(self.firebase, "db", None):
            try:
                from google.cloud import firestore

                repair_data["timestamp_server"] = firestore.SERVER_TIMESTAMP
                self.firebase.db.collection("repair_history").document(repair_id).set(
                    repair_data
                )
                return True, repair_id
            except Exception as exc:
                return False, str(exc)

        return False, "Firebase not available"

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(timezone.utc)
