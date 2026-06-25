"""
Lightweight validation for agent tool arguments before execution.
"""

from typing import Any, Dict, Optional

REQUIRED_FIELDS = {
    "tool_move_class": [
        "school_key",
        "batch_key",
        "from_day",
        "from_slot",
        "to_day",
        "to_slot",
    ],
    "tool_swap_classes": [
        "school_key",
        "batch_key1",
        "day1",
        "slot1",
        "batch_key2",
        "day2",
        "slot2",
    ],
    "tool_check_faculty_free": ["faculty_name", "day", "slot_key"],
    "tool_check_room_free": ["room_name", "day", "slot_key"],
    "tool_get_free_slots": ["faculty_name"],
    "tool_log_repair": ["action_type", "from_slot", "to_slot", "reason"],
    "tool_escalate": ["clash_description", "reason_unsolvable"],
    "tool_read_schedule": ["school_key", "batch_key"],
}


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def validate_tool_args(tool_name: str, tool_args: dict) -> Optional[str]:
    """Return an error message if validation fails, else None."""
    if tool_name not in REQUIRED_FIELDS:
        return None

    args = tool_args or {}
    missing = [field for field in REQUIRED_FIELDS[tool_name] if _is_blank(args.get(field))]
    if missing:
        return f"Missing or empty required fields for {tool_name}: {', '.join(missing)}"

    if tool_name in ("tool_move_class", "tool_swap_classes"):
        for key in ("from_day", "to_day", "day1", "day2"):
            if key in args and args[key] not in {
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            }:
                return f"Invalid day value in {tool_name}: {args[key]}"

    return None
