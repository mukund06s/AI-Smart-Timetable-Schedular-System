"""
Phase 4 edge-case handling: LLM retry, schedule revert, local backup.
"""

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MUTATING_TOOLS = frozenset(
    {"tool_move_class", "tool_swap_classes", "tool_apply_fix", "tool_place_class"}
)

LOCAL_SESSION_DIR = Path("research_output") / "local_agent_sessions"


def clash_count(schedule: dict, constraints: Optional[dict] = None) -> int:
    from utils.clash_analyzer import ClashAnalyzer

    analyzer = ClashAnalyzer()
    constraints = constraints or {}
    return len(
        analyzer.detect_all_clashes(
            schedule,
            existing_faculty_schedules=constraints.get("existing_faculty_schedules"),
            existing_room_schedules=constraints.get("existing_room_schedules"),
        )
    )


def restore_schedule(target: dict, snapshot: dict) -> None:
    """Replace target schedule contents with snapshot in place."""
    target.clear()
    target.update(copy.deepcopy(snapshot))


def execute_tool_with_revert_guard(
    tools,
    tool_name: str,
    tool_args: dict,
    schedule: dict,
    constraints: Optional[dict] = None,
) -> Tuple[dict, Optional[dict]]:
    """
    Execute a mutating tool and revert if clash count increases.

    Returns:
        (tool_result, revert_info or None)
    """
    if tool_name not in MUTATING_TOOLS:
        return tools.execute(tool_name, tool_args, schedule), None

    before_snapshot = copy.deepcopy(schedule)
    before_count = clash_count(schedule, constraints)
    result = tools.execute(tool_name, tool_args, schedule)

    if not result.get("success"):
        return result, None

    after_count = clash_count(schedule, constraints)
    if after_count <= before_count:
        return result, None

    restore_schedule(schedule, before_snapshot)
    revert_info = {
        "tool": tool_name,
        "input": tool_args,
        "clashes_before": before_count,
        "clashes_after": after_count,
        "message": "Reverted repair because it introduced new clashes",
    }
    result = {
        "success": False,
        "reverted": True,
        "message": revert_info["message"],
        "clashes_before": before_count,
        "clashes_after": after_count,
    }
    return result, revert_info


def call_llm_with_retry(client, max_retries: int = 3, retry_delay: float = 0.5, **kwargs):
    """Call Anthropic messages.create with retry on transient failures."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                break
            time.sleep(retry_delay * (2 ** attempt))
    raise last_error


def save_local_session_backup(session_payload: dict, session_id: str) -> str:
    """Persist agent session locally when Firebase write fails."""
    LOCAL_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCAL_SESSION_DIR / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(session_payload, handle, indent=2, default=str)
    return str(path)


def save_local_repair_backup(repair_payload: dict, repair_id: str) -> str:
    """Persist repair history locally when Firebase write fails."""
    backup_dir = LOCAL_SESSION_DIR / "repair_history"
    backup_dir.mkdir(parents=True, exist_ok=True)
    path = backup_dir / f"{repair_id}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(repair_payload, handle, indent=2, default=str)
    return str(path)
