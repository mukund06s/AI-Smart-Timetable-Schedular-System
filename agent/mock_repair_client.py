"""
Mock Anthropic client that auto-generates repair tool calls from detected clashes.
Used for Phase 3 stress testing without a live API key.
"""

import copy
from typing import Any, List, Optional

from utils.clash_analyzer import ClashAnalyzer


class MockToolUseBlock:
    def __init__(self, block_id, name, input_data):
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = input_data


class MockTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class MockAnthropicResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class AutoRepairMockMessages:
    """Generates tool_move_class / tool_escalate steps from the live schedule."""

    def __init__(self, schedule: dict, constraints: Optional[dict] = None, max_moves: int = 20):
        self.schedule = schedule
        self.constraints = constraints or {}
        self.max_moves = max_moves
        self.call_count = 0
        self.tool_stats = {}
        self.analyzer = ClashAnalyzer()

    def create(self, **kwargs):
        self.call_count += 1
        clashes = self.analyzer.detect_all_clashes(
            self.schedule,
            existing_faculty_schedules=self.constraints.get("existing_faculty_schedules"),
            existing_room_schedules=self.constraints.get("existing_room_schedules"),
        )
        if not clashes or self.call_count > self.max_moves:
            return MockAnthropicResponse([MockTextBlock("Repair complete.")], "end_turn")

        clash = clashes[0]
        move_args = self._build_move_for_clash(clash)
        if move_args:
            self._track_tool("tool_move_class")
            return MockAnthropicResponse(
                [
                    MockTextBlock(
                        f"Fixing {clash.get('type')} for "
                        f"{clash.get('faculty') or clash.get('room')}"
                    ),
                    MockToolUseBlock(
                        f"tool_{self.call_count}",
                        "tool_move_class",
                        move_args,
                    ),
                ],
                "tool_use",
            )

        self._track_tool("tool_escalate")
        return MockAnthropicResponse(
            [
                MockToolUseBlock(
                    f"tool_{self.call_count}",
                    "tool_escalate",
                    {
                        "clash_description": clash.get("details", clash.get("type", "")),
                        "reason_unsolvable": "No valid move found in mock auto-repair",
                    },
                )
            ],
            "tool_use",
        )

    def _build_move_for_clash(self, clash: dict) -> Optional[dict]:
        locations = clash.get("locations") or clash.get("bookings") or clash.get("sections") or []
        if len(locations) < 1:
            return None
        target = locations[-1] if len(locations) > 1 else locations[0]
        school = target.get("school", "BTECH")
        batch = target.get("batch")
        if not batch:
            return None

        time_field = clash.get("time", "")
        if " at " not in time_field:
            return None
        day, slot = time_field.split(" at ", 1)

        free_slots = self.analyzer.get_free_slots_for_faculty(
            self.schedule,
            clash.get("faculty", ""),
            existing_faculty_schedules=self.constraints.get("existing_faculty_schedules"),
            existing_room_schedules=self.constraints.get("existing_room_schedules"),
        )
        if not free_slots:
            return None

        candidate = next(
            (item for item in free_slots if item.get("batch") == batch),
            free_slots[0],
        )
        return {
            "school_key": school,
            "batch_key": batch,
            "from_day": day.strip(),
            "from_slot": slot.strip(),
            "to_day": candidate["day"],
            "to_slot": candidate["slot_key"],
        }

    def _track_tool(self, tool_name: str) -> None:
        self.tool_stats[tool_name] = self.tool_stats.get(tool_name, 0) + 1


class AutoRepairMockClient:
    def __init__(self, schedule: dict, constraints: Optional[dict] = None, max_moves: int = 20):
        self.messages = AutoRepairMockMessages(schedule, constraints, max_moves=max_moves)
        self.tool_stats = self.messages.tool_stats
