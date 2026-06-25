"""
Agent session memory and Firebase persistence for repair sessions.
"""

import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.firebase_ops import AgentFirebaseOps


class AgentMemory:
    """Stores the agent's session state across multiple turns."""

    def __init__(
        self,
        schedule: dict,
        clashes: List[dict],
        timetable_key: str = "",
        program: str = "",
        semester: Optional[int] = None,
    ):
        self.session_id = str(uuid.uuid4())
        self.timetable_key = timetable_key
        self.program = program
        self.semester = semester
        self.original_schedule = copy.deepcopy(schedule)
        self.current_schedule = schedule
        self.initial_clashes = copy.deepcopy(clashes)
        self.repairs_applied: List[dict] = []
        self.escalations: List[dict] = []
        self.turns_taken = 0
        self.clashes_found = len(clashes)
        self.clashes_fixed = 0
        self.conversation: List[dict] = []
        self.status = "in_progress"
        self.started_at = AgentFirebaseOps.utc_now()
        self.ended_at: Optional[datetime] = None
        self.firebase_saved = False
        self.local_backup_path: Optional[str] = None
        self.reverted_repairs: List[dict] = []
        self.llm_error: Optional[str] = None

    def log_action(self, tool_name: str, tool_input: dict, result: dict) -> None:
        """Record a tool invocation in session memory."""
        self.turns_taken += 1
        entry = {
            "turn": self.turns_taken,
            "tool": tool_name,
            "input": tool_input,
            "result": result,
            "timestamp": AgentFirebaseOps.utc_now().isoformat(),
        }
        self.conversation.append(entry)

        if tool_name == "tool_log_repair" and result.get("logged"):
            self.repairs_applied.append(
                {
                    "log_id": result.get("log_id"),
                    "action_type": tool_input.get("action_type"),
                    "from_slot": tool_input.get("from_slot"),
                    "to_slot": tool_input.get("to_slot"),
                    "reason": tool_input.get("reason"),
                    "success": tool_input.get("result", True),
                }
            )
            if tool_input.get("result", True):
                self.clashes_fixed += 1

        if tool_name == "tool_escalate" and result.get("flagged"):
            self.escalations.append(
                {
                    "clash_description": tool_input.get("clash_description"),
                    "reason_unsolvable": tool_input.get("reason_unsolvable"),
                    "log_id": result.get("log_id"),
                }
            )

    def record_revert(self, revert_info: dict) -> None:
        self.reverted_repairs.append(revert_info)

    def save_local_backup(self) -> str:
        from agent.edge_cases import save_local_session_backup

        payload = {
            "session_id": self.session_id,
            "timetable_key": self.timetable_key,
            "program": self.program,
            "semester": self.semester,
            "started_at": self.started_at.isoformat(),
            "ended_at": (self.ended_at or AgentFirebaseOps.utc_now()).isoformat(),
            "status": self.status,
            "clashes_found": self.clashes_found,
            "clashes_fixed": self.clashes_fixed,
            "turns_used": self.turns_taken,
            "conversation_log": self.conversation,
            "repairs_applied": self.repairs_applied,
            "escalations": self.escalations,
            "reverted_repairs": self.reverted_repairs,
            "llm_error": self.llm_error,
        }
        self.local_backup_path = save_local_session_backup(payload, self.session_id)
        return self.local_backup_path

    def log_llm_turn(self, turn_number: int, response_content: Any) -> None:
        """Store raw LLM response blocks for audit trail."""
        self.conversation.append(
            {
                "turn": turn_number,
                "type": "llm_response",
                "content": _serialize_response_content(response_content),
                "timestamp": AgentFirebaseOps.utc_now().isoformat(),
            }
        )

    def save_to_firebase(self, firebase_manager) -> tuple:
        """Saves this session to /agent_sessions/{session_id}."""
        ops = AgentFirebaseOps(firebase_manager)
        payload = {
            "session_id": self.session_id,
            "timetable_key": self.timetable_key,
            "program": self.program,
            "semester": self.semester,
            "started_at": self.started_at,
            "ended_at": self.ended_at or AgentFirebaseOps.utc_now(),
            "status": self.status,
            "clashes_found": self.clashes_found,
            "clashes_fixed": self.clashes_fixed,
            "turns_used": self.turns_taken,
            "conversation_log": self.conversation,
            "repairs_applied": self.repairs_applied,
            "escalations": self.escalations,
        }
        success, result = ops.save_agent_session(self.session_id, payload)
        if success:
            self.firebase_saved = True
            return True, result

        self.local_backup_path = self.save_local_backup()
        return False, self.local_backup_path

    def get_repair_summary(self) -> dict:
        """Returns human-readable summary for display."""
        end_time = self.ended_at or AgentFirebaseOps.utc_now()
        elapsed = (end_time - self.started_at).total_seconds()

        return {
            "session_id": self.session_id,
            "status": self.status,
            "clashes_found": self.clashes_found,
            "clashes_fixed": self.clashes_fixed,
            "escalated": len(self.escalations),
            "turns_used": self.turns_taken,
            "elapsed_seconds": round(elapsed, 2),
            "repairs_applied": self.repairs_applied,
            "escalations": self.escalations,
            "tool_call_counts": self._tool_call_counts(),
            "conversation_log": self.conversation,
            "firebase_saved": self.firebase_saved,
            "local_backup_path": self.local_backup_path,
            "reverted_repairs": self.reverted_repairs,
            "llm_error": self.llm_error,
        }

    def _tool_call_counts(self) -> dict:
        counts = {}
        for entry in self.conversation:
            tool = entry.get("tool")
            if tool:
                counts[tool] = counts.get(tool, 0) + 1
        return counts

    def get_summary(self) -> dict:
        """Alias used by TimetableAgent."""
        return self.get_repair_summary()


def _serialize_response_content(content: Any) -> Any:
    if isinstance(content, list):
        serialized = []
        for block in content:
            if hasattr(block, "model_dump"):
                serialized.append(block.model_dump())
            elif hasattr(block, "type"):
                serialized.append(
                    {
                        "type": getattr(block, "type", None),
                        "text": getattr(block, "text", None),
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                    }
                )
            else:
                serialized.append(str(block))
        return serialized
    return str(content)
