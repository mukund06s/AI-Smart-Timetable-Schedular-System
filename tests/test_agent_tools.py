"""
Unit tests for Agent tool registry — one test per blueprint tool.
"""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools import ToolRegistry
from tests.agent_test_helpers import MockFirebase, build_sample_schedule


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.firebase = MockFirebase()
        self.tools = ToolRegistry(self.firebase)
        self.schedule = build_sample_schedule(with_faculty_clash=True)
        self.tools.bind_context(
            schedule=self.schedule,
            constraints={"subjects": []},
            session_id="test-session",
        )

    def test_tool_1_read_schedule(self):
        result = self.tools.execute(
            "tool_read_schedule",
            {
                "school_key": "BTECH",
                "batch_key": "Sem_2_Section_A",
                "day": "Monday",
            },
            self.schedule,
        )
        self.assertTrue(result["success"])
        self.assertIn("schedule_dict", result)
        self.assertIn("09:00-10:00", result["schedule_dict"]["Monday"])

    def test_tool_2_read_clashes(self):
        result = self.tools.execute("tool_read_clashes", {}, self.schedule)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["clash_count"], 1)
        clash = result["clashes"][0]
        self.assertIn("type", clash)
        self.assertIn("time", clash)
        self.assertIn("description", clash)

    def test_tool_3_check_faculty_free(self):
        busy = self.tools.execute(
            "tool_check_faculty_free",
            {
                "faculty_name": "Dr. Mehta",
                "day": "Monday",
                "slot_key": "09:00-10:00",
            },
            self.schedule,
        )
        free = self.tools.execute(
            "tool_check_faculty_free",
            {
                "faculty_name": "Dr. Mehta",
                "day": "Tuesday",
                "slot_key": "10:00-11:00",
            },
            self.schedule,
        )
        self.assertFalse(busy["is_free"])
        self.assertIn("other_class", busy)
        self.assertTrue(free["is_free"])

    def test_tool_4_check_room_free(self):
        busy = self.tools.execute(
            "tool_check_room_free",
            {
                "room_name": "LH-101",
                "day": "Monday",
                "slot_key": "09:00-10:00",
                "school_key": "BTECH",
            },
            self.schedule,
        )
        free = self.tools.execute(
            "tool_check_room_free",
            {
                "room_name": "LH-102",
                "day": "Tuesday",
                "slot_key": "10:00-11:00",
                "school_key": "BTECH",
            },
            self.schedule,
        )
        self.assertFalse(busy["is_free"])
        self.assertTrue(free["is_free"])

    def test_tool_5_get_free_slots(self):
        result = self.tools.execute(
            "tool_get_free_slots",
            {"faculty_name": "Dr. Mehta", "day": "Tuesday"},
            self.schedule,
        )
        self.assertTrue(result["success"])
        self.assertGreater(result["count"], 0)
        slot = result["free_slots"][0]
        self.assertIn("day", slot)
        self.assertIn("slot_key", slot)
        self.assertIn("available_rooms", slot)
        self.assertIsInstance(slot["available_rooms"], list)

    def test_tool_6_move_class(self):
        result = self.tools.execute(
            "tool_move_class",
            {
                "school_key": "BTECH",
                "batch_key": "Sem_2_Section_B",
                "from_day": "Monday",
                "from_slot": "09:00-10:00",
                "to_day": "Tuesday",
                "to_slot": "10:00-11:00",
            },
            self.schedule,
        )
        self.assertTrue(result["success"])
        self.assertIn("message", result)

    def test_tool_7_swap_classes(self):
        schedule = copy.deepcopy(self.schedule)
        self.tools.bind_context(schedule=schedule, constraints={}, session_id="swap-test")
        result = self.tools.execute(
            "tool_swap_classes",
            {
                "school_key": "BTECH",
                "batch_key1": "Sem_2_Section_A",
                "day1": "Monday",
                "slot1": "09:00-10:00",
                "batch_key2": "Sem_2_Section_A",
                "day2": "Monday",
                "slot2": "11:00-12:00",
            },
            schedule,
        )
        self.assertTrue(result["success"])

    def test_tool_8_apply_fix_move(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        self.tools.bind_context(schedule=schedule, constraints={}, session_id="apply-test")
        result = self.tools.execute(
            "tool_apply_fix",
            {
                "action_type": "move",
                "school_key": "BTECH",
                "batch_key": "Sem_2_Section_B",
                "from_day": "Monday",
                "from_slot": "09:00-10:00",
                "to_day": "Tuesday",
                "to_slot": "09:00-10:00",
            },
            schedule,
        )
        self.assertTrue(result["success"])

    def test_tool_8_apply_fix_swap(self):
        schedule = copy.deepcopy(self.schedule)
        self.tools.bind_context(schedule=schedule, constraints={}, session_id="apply-swap")
        result = self.tools.execute(
            "tool_apply_fix",
            {
                "action_type": "swap",
                "school_key": "BTECH",
                "batch_key": "Sem_2_Section_A",
                "from_day": "Monday",
                "from_slot": "09:00-10:00",
                "to_day": "Monday",
                "to_slot": "11:00-12:00",
            },
            schedule,
        )
        self.assertTrue(result["success"])

    def test_tool_9_verify_schedule(self):
        result = self.tools.execute("tool_verify_schedule", {}, self.schedule)
        self.assertTrue(result["success"])
        self.assertIn("clash_count", result)
        self.assertIn("clashes", result)
        self.assertIn("lecture_violations", result)

    def test_tool_10_log_repair(self):
        result = self.tools.execute(
            "tool_log_repair",
            {
                "action_type": "move",
                "from_slot": {"day": "Monday", "slot": "09:00-10:00", "batch_key": "Sem_2_Section_B"},
                "to_slot": {"day": "Tuesday", "slot": "10:00-11:00", "batch_key": "Sem_2_Section_B"},
                "reason": "Resolved faculty clash",
                "result": True,
                "clash_type": "faculty",
                "faculty_or_room": "Dr. Mehta",
            },
            self.schedule,
        )
        self.assertTrue(result["logged"])
        self.assertIn("log_id", result)
        self.assertIn(f"repair_history/{result['log_id']}", self.firebase.saved)

    def test_tool_11_escalate(self):
        result = self.tools.execute(
            "tool_escalate",
            {
                "clash_description": "Dr. Mehta clash",
                "reason_unsolvable": "No free slot",
            },
            self.schedule,
        )
        self.assertTrue(result["flagged"])
        self.assertTrue(result["logged"])
        self.assertIn(f"repair_history/{result['log_id']}", self.firebase.saved)

    def test_get_all_tools_matches_blueprint(self):
        tools = self.tools.get_all_tools()
        names = [tool["name"] for tool in tools]
        expected = [
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
        ]
        for tool_name in expected:
            self.assertIn(tool_name, names)
        self.assertIn("tool_escalate", names)


if __name__ == "__main__":
    unittest.main()
