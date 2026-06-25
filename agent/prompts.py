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
Your job is to fix scheduling conflicts (clashes) by moving or swapping classes.

[CONTEXT]
- Faculty clashes: Same teacher in 2 places at the same time
- Room clashes: Same room used by 2 different classes
- Cross-semester clashes: Teacher busy in another semester's timetable

[CONSTRAINTS YOU MUST RESPECT]
1. A faculty cannot teach more than 2 classes at 9AM per week
2. Lunch slot (1:00-1:50PM) cannot be used for lectures
3. Lab sessions must be 2 consecutive hours
4. Theory classes: 1 hour each
5. Do not create new clashes while fixing existing ones

[TOOLS]
You have access to: {tool_names}

[REASONING STYLE]
Think step by step:
1. Read the clash report
2. Understand WHY it happened
3. Find the best slot to move the clashing class to
4. Verify target slot is free (faculty + room)
5. Apply the fix
6. Verify no new clash created
7. Move to next clash

[OUTPUT FORMAT]
Always structure your response as:
THOUGHT: [your reasoning]
ACTION: [tool name + arguments]
Then wait for OBSERVATION before proceeding."""


def build_initial_user_message(clashes: List[dict]) -> str:
    """Format the opening user message for the repair loop."""
    lines = []
    for index, clash in enumerate(clashes, 1):
        entity = clash.get("faculty") or clash.get("room") or "Unknown"
        lines.append(
            f"{index}. [{clash.get('type', 'Unknown')}] {entity} "
            f"at {clash.get('time', '')} — {clash.get('details', clash.get('description', ''))}"
        )

    clash_summary = "\n".join(lines) if lines else "No clashes listed."
    return (
        f"Please repair the following {len(clashes)} clashes "
        f"in the college timetable:\n\n{clash_summary}\n\n"
        f"Use your tools to fix each clash one by one."
    )
