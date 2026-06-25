"""
Tool registry for the Agentic Timetable Repair Agent.
All tools exposed to the LLM with Anthropic-compatible schemas.
"""

import copy
import uuid
from typing import Any, Dict, List, Optional

from agent.firebase_ops import AgentFirebaseOps
from utils.clash_analyzer import ClashAnalyzer


class ToolRegistry:
    """All tools the Agent can call."""

    TOOL_NAMES = [
        "tool_read_schedule",
        "tool_read_clashes",
        "tool_move_class",
        "tool_swap_classes",
        "tool_check_faculty_free",
        "tool_check_room_free",
        "tool_get_free_slots",
        "tool_apply_fix",
        "tool_verify_schedule",
        "tool_log_repair",
        "tool_escalate",
    ]

    @classmethod
    def get_tool_names_for_prompt(cls) -> List[str]:
        return list(cls.TOOL_NAMES)

    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
        self.firebase_ops = AgentFirebaseOps(firebase_manager)
        self.analyzer = ClashAnalyzer()
        self._schedule: dict = {}
        self._constraints: dict = {}
        self._memory = None
        self._session_id = ""

    def bind_context(
        self,
        schedule: dict,
        constraints: Optional[dict] = None,
        memory=None,
        session_id: str = "",
    ) -> None:
        self._schedule = schedule
        self._constraints = constraints or {}
        self._memory = memory
        self._session_id = session_id or (memory.session_id if memory else "")

    def get_all_tools(self) -> List[dict]:
        """Returns tool definitions in Anthropic format."""
        return [
            self._schema(
                "tool_read_schedule",
                "Read full timetable for a section. Use before making changes.",
                {
                    "school_key": {"type": "string", "description": "School/program key"},
                    "batch_key": {"type": "string", "description": "Batch key e.g. Sem_2_Section_A"},
                    "day": {
                        "type": "string",
                        "description": "Optional day filter e.g. Monday",
                    },
                },
                required=["school_key", "batch_key"],
            ),
            self._schema(
                "tool_read_clashes",
                "Get structured list of all current clashes in the schedule.",
                {
                    "schedule": {
                        "type": "object",
                        "description": "Optional schedule override. Defaults to current schedule.",
                    }
                },
                required=[],
            ),
            self._schema(
                "tool_check_faculty_free",
                "Check whether a faculty member is free at a specific day and slot.",
                {
                    "faculty_name": {"type": "string"},
                    "day": {"type": "string"},
                    "slot_key": {"type": "string", "description": "Format: 09:00-10:00"},
                },
                required=["faculty_name", "day", "slot_key"],
            ),
            self._schema(
                "tool_check_room_free",
                "Check whether a room is free at a specific day and slot.",
                {
                    "room_name": {"type": "string"},
                    "day": {"type": "string"},
                    "slot_key": {"type": "string"},
                    "school_key": {"type": "string"},
                },
                required=["room_name", "day", "slot_key"],
            ),
            self._schema(
                "tool_get_free_slots",
                "Find free slots where a faculty member can be scheduled.",
                {
                    "faculty_name": {"type": "string"},
                    "day": {
                        "type": "string",
                        "description": "Optional day filter",
                    },
                },
                required=["faculty_name"],
            ),
            self._schema(
                "tool_move_class",
                "Move a class from one day/slot to another day/slot.",
                {
                    "school_key": {"type": "string"},
                    "batch_key": {"type": "string"},
                    "from_day": {"type": "string"},
                    "from_slot": {"type": "string"},
                    "to_day": {"type": "string"},
                    "to_slot": {"type": "string"},
                },
                required=[
                    "school_key",
                    "batch_key",
                    "from_day",
                    "from_slot",
                    "to_day",
                    "to_slot",
                ],
            ),
            self._schema(
                "tool_swap_classes",
                "Swap two classes between their slots.",
                {
                    "school_key": {"type": "string"},
                    "batch_key1": {"type": "string"},
                    "day1": {"type": "string"},
                    "slot1": {"type": "string"},
                    "batch_key2": {"type": "string"},
                    "day2": {"type": "string"},
                    "slot2": {"type": "string"},
                },
                required=[
                    "school_key",
                    "batch_key1",
                    "day1",
                    "slot1",
                    "batch_key2",
                    "day2",
                    "slot2",
                ],
            ),
            self._schema(
                "tool_apply_fix",
                "Apply a generic repair action. Supports move or swap.",
                {
                    "action_type": {
                        "type": "string",
                        "enum": ["move", "swap"],
                    },
                    "school_key": {"type": "string"},
                    "batch_key": {"type": "string"},
                    "from_day": {"type": "string"},
                    "from_slot": {"type": "string"},
                    "to_day": {"type": "string"},
                    "to_slot": {"type": "string"},
                    "batch_key2": {"type": "string"},
                    "day2": {"type": "string"},
                    "slot2": {"type": "string"},
                },
                required=["action_type", "school_key", "batch_key"],
            ),
            self._schema(
                "tool_verify_schedule",
                "Run full clash detection after a fix and return remaining issues.",
                {},
                required=[],
            ),
            self._schema(
                "tool_log_repair",
                "Save a repair action to Firebase repair history.",
                {
                    "action_type": {"type": "string"},
                    "from_slot": {"type": "object"},
                    "to_slot": {"type": "object"},
                    "reason": {"type": "string"},
                    "result": {"type": "boolean"},
                    "clash_type": {"type": "string"},
                    "faculty_or_room": {"type": "string"},
                },
                required=["action_type", "from_slot", "to_slot", "reason"],
            ),
            self._schema(
                "tool_escalate",
                "Mark a clash as unresolvable and flag it for manual review.",
                {
                    "clash_description": {"type": "string"},
                    "reason_unsolvable": {"type": "string"},
                },
                required=["clash_description", "reason_unsolvable"],
            ),
        ]

    def execute(
        self, tool_name: str, tool_args: dict, schedule: Optional[dict] = None
    ) -> dict:
        """Route a tool call to its handler."""
        if schedule is not None:
            self._schedule = schedule

        handlers = {
            "tool_read_schedule": self._tool_read_schedule,
            "tool_read_clashes": self._tool_read_clashes,
            "tool_check_faculty_free": self._tool_check_faculty_free,
            "tool_check_room_free": self._tool_check_room_free,
            "tool_get_free_slots": self._tool_get_free_slots,
            "tool_move_class": self._tool_move_class,
            "tool_swap_classes": self._tool_swap_classes,
            "tool_apply_fix": self._tool_apply_fix,
            "tool_verify_schedule": self._tool_verify_schedule,
            "tool_log_repair": self._tool_log_repair,
            "tool_escalate": self._tool_escalate,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        from agent.input_validation import validate_tool_args

        validation_error = validate_tool_args(tool_name, tool_args or {})
        if validation_error:
            return {"success": False, "error": validation_error}

        return handler(tool_args or {})

    def _tool_read_schedule(self, args: dict) -> dict:
        school_key = args.get("school_key")
        batch_key = args.get("batch_key")
        day = args.get("day")

        if school_key not in self._schedule:
            return {"success": False, "error": f"Unknown school_key: {school_key}"}
        if batch_key not in self._schedule[school_key]:
            return {"success": False, "error": f"Unknown batch_key: {batch_key}"}

        batch_schedule = self._schedule[school_key][batch_key]
        if day:
            if day not in batch_schedule:
                return {"success": False, "error": f"Unknown day: {day}"}
            schedule_dict = {day: copy.deepcopy(batch_schedule[day])}
        else:
            schedule_dict = copy.deepcopy(batch_schedule)

        return {
            "success": True,
            "school_key": school_key,
            "batch_key": batch_key,
            "schedule_dict": schedule_dict,
        }

    def _tool_read_clashes(self, args: dict) -> dict:
        schedule = args.get("schedule") or self._schedule
        existing_faculty = self._constraints.get("existing_faculty_schedules")
        existing_room = self._constraints.get("existing_room_schedules")
        clashes = self.analyzer.detect_all_clashes(
            schedule,
            existing_faculty_schedules=existing_faculty,
            existing_room_schedules=existing_room,
        )
        return {
            "success": True,
            "clash_count": len(clashes),
            "clashes": clashes,
        }

    def _tool_check_faculty_free(self, args: dict) -> dict:
        faculty_name = args["faculty_name"]
        day = args["day"]
        slot_key = args["slot_key"]

        assignments = self.analyzer.get_faculty_assignments_at(
            self._schedule, faculty_name, day, slot_key
        )
        existing_faculty = self._constraints.get("existing_faculty_schedules", {})
        key = f"{day}_{slot_key}"
        cross_semester = (
            faculty_name in existing_faculty and key in existing_faculty[faculty_name]
        )

        if assignments or cross_semester:
            other_class = assignments[0] if assignments else {
                "source": "cross_semester",
                "details": existing_faculty.get(faculty_name, {}).get(key),
            }
            return {
                "is_free": False,
                "other_class": other_class,
            }

        return {"is_free": True}

    def _tool_check_room_free(self, args: dict) -> dict:
        room_name = args["room_name"]
        day = args["day"]
        slot_key = args["slot_key"]

        assignments = self.analyzer.get_room_assignments_at(
            self._schedule, room_name, day, slot_key
        )
        existing_room = self._constraints.get("existing_room_schedules", {})
        key = f"{day}_{slot_key}"
        cross_semester = room_name in existing_room and key in existing_room[room_name]

        if assignments or cross_semester:
            other_class = assignments[0] if assignments else {
                "source": "cross_semester",
                "details": existing_room.get(room_name, {}).get(key),
            }
            return {
                "is_free": False,
                "other_class": other_class,
            }

        return {"is_free": True}

    def _tool_get_free_slots(self, args: dict) -> dict:
        faculty_name = args["faculty_name"]
        day = args.get("day")
        existing_faculty = self._constraints.get("existing_faculty_schedules")
        existing_room = self._constraints.get("existing_room_schedules")
        free_slots = self.analyzer.get_free_slots_for_faculty(
            self._schedule,
            faculty_name,
            day=day,
            existing_faculty_schedules=existing_faculty,
            existing_room_schedules=existing_room,
        )
        formatted = [
            {
                "day": slot_info["day"],
                "slot_key": slot_info["slot_key"],
                "available_rooms": slot_info["available_rooms"],
            }
            for slot_info in free_slots
        ]
        return {
            "success": True,
            "faculty_name": faculty_name,
            "free_slots": formatted,
            "count": len(formatted),
        }

    def _tool_move_class(self, args: dict) -> dict:
        school_key = args["school_key"]
        batch_key = args["batch_key"]
        from_day = args["from_day"]
        from_slot = args["from_slot"]
        to_day = args["to_day"]
        to_slot = args["to_slot"]

        try:
            source = self._schedule[school_key][batch_key][from_day][from_slot]
        except KeyError as exc:
            return {"success": False, "message": f"Source slot not found: {exc}"}

        if source is None or (
            isinstance(source, dict)
            and source.get("type") in self.analyzer.SKIP_TYPES
        ):
            return {"success": False, "message": "Source slot has no movable class"}

        target = self._schedule[school_key][batch_key][to_day][to_slot]
        if self.analyzer._slot_has_teaching_class(target):
            return {"success": False, "message": "Target slot is not empty"}

        class_info = copy.deepcopy(source)
        self._schedule[school_key][batch_key][from_day][from_slot] = None
        self._schedule[school_key][batch_key][to_day][to_slot] = class_info

        return {
            "success": True,
            "message": (
                f"Moved class from {from_day} {from_slot} to {to_day} {to_slot}"
            ),
            "moved_class": class_info,
        }

    def _tool_swap_classes(self, args: dict) -> dict:
        school_key = args["school_key"]
        batch_key1 = args["batch_key1"]
        day1 = args["day1"]
        slot1 = args["slot1"]
        batch_key2 = args["batch_key2"]
        day2 = args["day2"]
        slot2 = args["slot2"]

        try:
            class_a = self._schedule[school_key][batch_key1][day1][slot1]
            class_b = self._schedule[school_key][batch_key2][day2][slot2]
        except KeyError as exc:
            return {"success": False, "message": f"Swap slot not found: {exc}"}

        self._schedule[school_key][batch_key1][day1][slot1] = class_b
        self._schedule[school_key][batch_key2][day2][slot2] = class_a

        return {
            "success": True,
            "message": f"Swapped {day1} {slot1} with {day2} {slot2}",
        }

    def _tool_apply_fix(self, args: dict) -> dict:
        action_type = args.get("action_type")
        if action_type == "move":
            return self._tool_move_class(args)
        if action_type == "swap":
            return self._tool_swap_classes(
                {
                    "school_key": args["school_key"],
                    "batch_key1": args["batch_key"],
                    "day1": args.get("from_day"),
                    "slot1": args.get("from_slot"),
                    "batch_key2": args.get("batch_key2", args["batch_key"]),
                    "day2": args.get("to_day"),
                    "slot2": args.get("to_slot"),
                }
            )
        return {"success": False, "message": f"Unsupported action_type: {action_type}"}

    def _tool_verify_schedule(self, args: dict) -> dict:
        schedule = args.get("schedule") or self._schedule
        existing_faculty = self._constraints.get("existing_faculty_schedules")
        existing_room = self._constraints.get("existing_room_schedules")
        clashes = self.analyzer.detect_all_clashes(
            schedule,
            existing_faculty_schedules=existing_faculty,
            existing_room_schedules=existing_room,
        )
        lecture_violations = self.analyzer.count_lecture_violations(
            schedule, self._constraints
        )
        return {
            "success": True,
            "clash_count": len(clashes),
            "clashes": clashes,
            "lecture_violations": lecture_violations,
        }

    def _tool_log_repair(self, args: dict) -> dict:
        repair_id = str(uuid.uuid4())
        payload = {
            "repair_id": repair_id,
            "session_id": self._session_id,
            "timestamp": AgentFirebaseOps.utc_now().isoformat(),
            "action_type": args.get("action_type"),
            "clash_type": args.get("clash_type", ""),
            "faculty_or_room": args.get("faculty_or_room", ""),
            "from_slot": args.get("from_slot"),
            "to_slot": args.get("to_slot"),
            "reason": args.get("reason"),
            "success": bool(args.get("result", True)),
        }

        success, result = self.firebase_ops.save_repair_history(repair_id, payload)
        if success:
            return {"logged": True, "log_id": repair_id}

        from agent.edge_cases import save_local_repair_backup

        local_path = save_local_repair_backup(payload, repair_id)
        return {
            "logged": True,
            "log_id": repair_id,
            "local_backup": local_path,
            "firebase_error": result,
        }

    def _tool_escalate(self, args: dict) -> dict:
        repair_id = str(uuid.uuid4())
        payload = {
            "repair_id": repair_id,
            "session_id": self._session_id,
            "timestamp": AgentFirebaseOps.utc_now().isoformat(),
            "action_type": "escalate",
            "clash_type": args.get("clash_type", ""),
            "faculty_or_room": args.get("faculty_or_room", ""),
            "from_slot": args.get("from_slot"),
            "to_slot": args.get("to_slot"),
            "reason": args.get("reason_unsolvable", args.get("clash_description", "")),
            "success": False,
            "clash_description": args.get("clash_description"),
            "reason_unsolvable": args.get("reason_unsolvable"),
        }
        logged, log_id = self.firebase_ops.save_repair_history(repair_id, payload)
        return {
            "flagged": True,
            "message": "Clash escalated for manual review",
            "clash_description": args.get("clash_description"),
            "reason_unsolvable": args.get("reason_unsolvable"),
            "logged": logged,
            "log_id": log_id if logged else repair_id,
        }

    @staticmethod
    def _schema(
        name: str,
        description: str,
        properties: dict,
        required: List[str],
    ) -> dict:
        return {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
