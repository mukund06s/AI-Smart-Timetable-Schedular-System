"""
Deterministic fallback for incomplete schedules (missing theory/lab/tutorial hours).
Generic — uses constraints.subjects, not hardcoded subject names.
"""

from typing import List, Optional

from agent.tools import ToolRegistry
from utils.clash_analyzer import ClashAnalyzer


def deterministic_gap_repair(
    schedule: dict,
    constraints: dict,
    gaps: Optional[List[dict]] = None,
    tools: Optional[ToolRegistry] = None,
) -> int:
    """
    Place missing sessions into empty slots where faculty and room are free.
    Returns the number of sessions successfully placed.
    """
    analyzer = ClashAnalyzer()
    gaps = gaps or analyzer.detect_scheduling_gaps(schedule, constraints)
    if not gaps:
        return 0

    registry = tools or ToolRegistry(None)
    registry.bind_context(schedule=schedule, constraints=constraints)

    existing_faculty = constraints.get("existing_faculty_schedules", {})
    existing_room = constraints.get("existing_room_schedules", {})
    placed_count = 0

    for gap in gaps:
        missing = int(gap.get("missing_hours", 1) or 1)
        faculty = gap.get("faculty", "TBD")
        preferred_room = (gap.get("room") or "").strip()
        school_key = gap.get("school_key") or next(iter(schedule), "")
        batch_key = gap.get("batch_key", "")
        class_type = gap.get("class_type", "Theory")
        lab_batch = gap.get("lab_batch", "")

        for _ in range(missing):
            free_slots = analyzer.get_free_slots_for_faculty(
                schedule,
                faculty,
                existing_faculty_schedules=existing_faculty,
                existing_room_schedules=existing_room,
            )

            placed = False
            for slot_info in free_slots:
                day = slot_info["day"]
                slot_key = slot_info["slot_key"]
                available_rooms = slot_info.get("available_rooms") or []

                room_candidates = []
                if preferred_room:
                    room_candidates.append(preferred_room)
                room_candidates.extend(
                    r for r in available_rooms if r not in room_candidates
                )
                if not room_candidates:
                    room_candidates = ["TBD"]

                for room_name in room_candidates:
                    args = {
                        "school_key": school_key,
                        "batch_key": batch_key,
                        "day": day,
                        "slot_key": slot_key,
                        "subject": gap["subject"],
                        "faculty": faculty,
                        "room": room_name,
                        "class_type": class_type,
                    }
                    if lab_batch:
                        args["lab_batch"] = lab_batch

                    result = registry.execute("tool_place_class", args, schedule)
                    if result.get("success"):
                        placed_count += 1
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                break

    return placed_count
