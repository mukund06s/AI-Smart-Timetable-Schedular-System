"""
System prompts for the Agentic Timetable Repair Agent.
"""

from typing import Any, Dict, List

from agent.tools import ToolRegistry


def build_system_prompt(constraints: Dict[str, Any] | None = None) -> str:
    """Build the system prompt injected into every agent repair session."""
    constraints = constraints or {}
    tool_names = ", ".join(ToolRegistry.get_tool_names_for_prompt())

    return f"""[ROLE]
You are an intelligent timetable repair agent for a college scheduling system.
Your job is to fix scheduling problems: clashes AND missing/incomplete subject assignments.

[CONTEXT]
- Faculty clashes: Same teacher in 2 places at the same time
- Room clashes: Same room used by 2 different classes
- Cross-semester clashes: Teacher busy in another semester's timetable
- Incomplete schedules: A subject has fewer sessions than required (e.g. missing Calculus Tutorial)

[CONSTRAINTS YOU MUST RESPECT]
1. A faculty cannot teach more than 2 classes at 9AM per week
2. Lunch slot cannot be used for lectures
3. Lab sessions must be 2 consecutive hours when scheduling 2-hour labs
4. Theory/Tutorial classes: 1 hour each
5. Do not create new clashes while fixing existing ones
6. Use assigned rooms from the room dataset when provided

[TOOLS]
You have access to: {tool_names}

For INCOMPLETE schedules (missing sessions):
- Use tool_read_scheduling_gaps to see what's missing
- Use tool_get_free_slots + tool_check_room_free to find valid slots
- Use tool_place_class to ADD a new session in an empty slot (do NOT use tool_move_class for missing classes)

For CLASHES:
- Use tool_move_class or tool_swap_classes to relocate existing classes

[REASONING STYLE]
Think step by step:
1. Read clashes and/or scheduling gaps
2. Prioritize incomplete required sessions (missing tutorials/labs/theory)
3. Find free slots (faculty + room available)
4. Apply fix with tool_place_class or tool_move_class
5. Verify with tool_verify_schedule (clash_count AND lecture_violations should decrease)
6. Continue until schedule is complete and clash-free

[OUTPUT FORMAT]
Always structure your response as:
THOUGHT: [your reasoning]
ACTION: [tool name + arguments]
Then wait for OBSERVATION before proceeding."""


def build_initial_user_message(
    clashes: List[dict], scheduling_gaps: List[dict] | None = None
) -> str:
    """Format the opening user message for the repair loop."""
    scheduling_gaps = scheduling_gaps or []
    lines = []

    if clashes:
        lines.append(f"=== CLASHES ({len(clashes)}) ===")
        for index, clash in enumerate(clashes, 1):
            entity = clash.get("faculty") or clash.get("room") or "Unknown"
            lines.append(
                f"{index}. [{clash.get('type', 'Unknown')}] {entity} "
                f"at {clash.get('time', '')} — {clash.get('details', clash.get('description', ''))}"
            )

    if scheduling_gaps:
        lines.append(f"\n=== INCOMPLETE SCHEDULING ({len(scheduling_gaps)}) ===")
        for index, gap in enumerate(scheduling_gaps, 1):
            lines.append(
                f"{index}. [{gap.get('type', 'IncompleteSchedule')}] "
                f"{gap.get('subject')} ({gap.get('class_type')}) Section {gap.get('section')} — "
                f"{gap.get('details', gap.get('description', ''))}. "
                f"Use batch_key={gap.get('batch_key')}, faculty={gap.get('faculty')}, "
                f"room={gap.get('room') or 'from dataset'}."
            )

    issue_summary = "\n".join(lines) if lines else "No issues listed."
    total = len(clashes) + len(scheduling_gaps)
    return (
        f"Please repair the following {total} timetable issue(s):\n\n{issue_summary}\n\n"
        f"Fix incomplete sessions with tool_place_class. Fix clashes with move/swap. "
        f"Verify after each fix."
    )
