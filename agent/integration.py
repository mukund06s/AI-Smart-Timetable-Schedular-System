"""
Shared integration layer between TimetableAgent and the Streamlit app pipeline.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.gap_repair import deterministic_gap_repair
from agent.rate_limiter import get_rate_limiter
from agent.timetable_agent import TimetableAgent
from utils.clash_analyzer import ClashAnalyzer
from utils.logging_config import get_logger

logger = get_logger(__name__)


def get_anthropic_api_key() -> str:
    """Resolve Gemini API key from Streamlit secrets or environment."""
    try:
        import streamlit as st

        key = st.secrets["agent"].get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def format_turn_log_entry(turn: int, response: Any) -> List[str]:
    """Convert an Anthropic response into blueprint-style log lines."""
    lines: List[str] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text" and getattr(block, "text", None):
            lines.append(f"Turn {turn} — THOUGHT:\n   \"{block.text}\"")
        elif block_type == "tool_use":
            lines.append(f"Turn {turn} — ACTION: {block.name}")
            lines.append(f"   {json.dumps(block.input, indent=2)}")
    return lines


def run_agentic_clash_repair(
    firebase_manager,
    genetic_algorithm,
    schedule: dict,
    clashes: List[dict],
    constraints: dict,
    program: str,
    semester: Optional[int],
    clash_detector,
    api_key: Optional[str] = None,
    llm_client: Optional[Any] = None,
    on_turn_callback: Optional[Callable[[int, Any], None]] = None,
    enable_fallback: bool = True,
    scheduling_gaps: Optional[List[dict]] = None,
) -> Tuple[dict, dict, List[dict], List[str]]:
    """
    Run TimetableAgent repair for clashes and/or incomplete schedules,
    then optionally fall back to deterministic gap repair and _intelligent_repair.

    Returns:
        (repaired_schedule, agent_summary, remaining_clashes, agent_log_lines)
    """
    analyzer = ClashAnalyzer()
    if scheduling_gaps is None:
        scheduling_gaps = analyzer.detect_scheduling_gaps(schedule, constraints)

    agent_config = {}
    if firebase_manager and hasattr(firebase_manager, "get_agent_config"):
        agent_config = firebase_manager.get_agent_config()

    timetable_key = f"{program}_Sem{semester}" if program and semester else "unknown"
    agent_log: List[str] = []
    summary: dict = {
        "status": "skipped",
        "clashes_found": len(clashes),
        "gaps_found": len(scheduling_gaps),
    }

    existing_faculty = constraints.get("existing_faculty_schedules", {})
    existing_room = constraints.get("existing_room_schedules", {})
    has_issues = bool(clashes) or bool(scheduling_gaps)

    def _combined_callback(turn: int, response: Any) -> None:
        agent_log.extend(format_turn_log_entry(turn, response))
        if on_turn_callback:
            on_turn_callback(turn, response)

    agent_enabled = agent_config.get("enabled", True)
    if agent_enabled and has_issues:
        rate_key = timetable_key or "default"
        allowed, rate_message = get_rate_limiter().allow(rate_key)
        if not allowed:
            logger.warning("Agent repair blocked by rate limit for key=%s", rate_key)
            agent_log.append(f"⚠️ {rate_message}")
            summary = {
                "status": "rate_limited",
                "clashes_found": len(clashes),
                "gaps_found": len(scheduling_gaps),
                "clashes_fixed": 0,
                "gaps_fixed": 0,
                "rate_limit_message": rate_message,
            }
        else:
            resolved_key = api_key if api_key is not None else get_anthropic_api_key()
            agent = TimetableAgent(
                firebase_manager=firebase_manager,
                llm_client=llm_client,
                api_key=resolved_key,
            )
            schedule, summary = agent.repair_schedule(
                schedule=schedule,
                clashes=clashes,
                scheduling_gaps=scheduling_gaps,
                constraints=constraints,
                on_turn_callback=_combined_callback,
                timetable_key=timetable_key,
                program=program or "",
                semester=semester,
            )
            summary.setdefault("gaps_found", len(scheduling_gaps))
    elif not has_issues:
        summary = {
            "status": "completed",
            "clashes_found": 0,
            "gaps_found": 0,
            "clashes_fixed": 0,
            "gaps_fixed": 0,
        }

    remaining_clashes = clash_detector.detect_all_clashes(
        schedule,
        existing_faculty_schedules=existing_faculty,
        existing_room_schedules=existing_room,
    )
    remaining_gaps = analyzer.detect_scheduling_gaps(schedule, constraints)

    should_gap_fallback = (
        enable_fallback
        and bool(remaining_gaps)
        and agent_config.get("fallback_to_random_repair", True)
    )
    if should_gap_fallback:
        agent_log.append(
            f"⚠️ Fallback: placing {len(remaining_gaps)} incomplete subject(s)..."
        )
        placed = deterministic_gap_repair(schedule, constraints, remaining_gaps)
        summary["gap_fallback_used"] = True
        summary["gaps_fixed"] = summary.get("gaps_fixed", 0) + placed
        agent_log.append(f"   └─ Gap fallback placed {placed} session(s)")
        remaining_gaps = analyzer.detect_scheduling_gaps(schedule, constraints)
        if not remaining_gaps and not remaining_clashes:
            summary["status"] = "completed"

    should_clash_fallback = (
        enable_fallback
        and bool(remaining_clashes)
        and agent_config.get("fallback_to_random_repair", True)
    )
    if not agent_enabled:
        should_clash_fallback = bool(remaining_clashes)
    elif summary.get("status") == "completed" and not remaining_clashes:
        should_clash_fallback = False
    elif summary.get("status") in (
        "failed",
        "partial",
        "max_turns_exceeded",
        "llm_failed",
        "rate_limited",
    ) and remaining_clashes:
        should_clash_fallback = True

    if summary.get("status") == "max_turns_exceeded":
        agent_log.append(
            f"⚠️ Agent exceeded max turns ({summary.get('turns_used', 0)}). "
            "Triggering graceful fallback..."
        )
    if summary.get("status") == "llm_failed":
        agent_log.append(
            f"⚠️ LLM API failed after retries: {summary.get('llm_error', 'unknown error')}. "
            "Triggering fallback..."
        )
    if summary.get("status") == "rate_limited":
        agent_log.append(
            f"⚠️ {summary.get('rate_limit_message', 'Agent rate limit reached')}. "
            "Triggering fallback..."
        )
    if summary.get("reverted_repairs"):
        agent_log.append(
            f"↩️ {len(summary['reverted_repairs'])} repair(s) reverted "
            "(introduced new clashes)."
        )
    if summary.get("local_backup_path") and not summary.get("firebase_saved", True):
        agent_log.append(
            f"💾 Firebase unavailable — session saved locally at "
            f"{summary['local_backup_path']}"
        )

    if should_clash_fallback and remaining_clashes and genetic_algorithm:
        agent_status = summary.get("status")
        summary["agent_status_before_fallback"] = agent_status
        agent_log.append(
            "⚠️ Fallback: running legacy _intelligent_repair rounds..."
        )
        for repair_i in range(25):
            genetic_algorithm._intelligent_repair(schedule, constraints)
            remaining_clashes = clash_detector.detect_all_clashes(
                schedule,
                existing_faculty_schedules=existing_faculty,
                existing_room_schedules=existing_room,
            )
            if not remaining_clashes:
                agent_log.append(
                    f"   └─ Fallback repair resolved all clashes in round {repair_i + 1} ✅"
                )
                break
        summary["fallback_used"] = True
        if not remaining_clashes and not remaining_gaps:
            summary["status"] = "completed"
        elif agent_status in ("llm_failed", "max_turns_exceeded"):
            summary["status"] = agent_status
        else:
            summary["status"] = "partial"

    summary["remaining_clashes"] = len(remaining_clashes)
    summary["remaining_gaps"] = len(remaining_gaps)
    return schedule, summary, remaining_clashes, agent_log


def categorize_clashes(clashes: List[dict]) -> Dict[str, int]:
    """Split clashes into faculty, room, and cross-semester counts."""
    faculty = 0
    room = 0
    cross = 0
    for clash in clashes:
        clash_type = clash.get("type", "")
        if "Cross-Semester" in clash_type:
            cross += 1
        elif "Faculty" in clash_type:
            faculty += 1
        elif "Room" in clash_type:
            room += 1
    return {"faculty": faculty, "room": room, "cross_semester": cross}
