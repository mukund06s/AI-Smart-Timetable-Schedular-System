"""
Firebase integration tests for agent sessions and repair history.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.firebase_ops import AgentFirebaseOps
from agent.memory import AgentMemory
from agent.tools import ToolRegistry
from tests.agent_test_helpers import MockFirebase, build_sample_schedule


class AgentFirebaseTests(unittest.TestCase):
    def setUp(self):
        self.firebase = MockFirebase()
        self.ops = AgentFirebaseOps(self.firebase)

    def test_save_agent_session_via_firebase_manager(self):
        memory = AgentMemory(
            schedule=build_sample_schedule(),
            clashes=[{"type": "Faculty Clash", "details": "test"}],
            timetable_key="BTECH_Sem2",
            program="BTECH",
            semester=2,
        )
        memory.status = "completed"
        success, session_id = memory.save_to_firebase(self.firebase)
        self.assertTrue(success)
        self.assertIn(f"agent_sessions/{session_id}", self.firebase.saved)

    def test_save_repair_history_via_tool(self):
        tools = ToolRegistry(self.firebase)
        schedule = build_sample_schedule(with_faculty_clash=True)
        tools.bind_context(schedule=schedule, session_id="firebase-session")
        result = tools.execute(
            "tool_log_repair",
            {
                "action_type": "move",
                "from_slot": {"day": "Monday", "slot": "09:00-10:00"},
                "to_slot": {"day": "Tuesday", "slot": "10:00-11:00"},
                "reason": "Firebase integration test",
                "result": True,
            },
            schedule,
        )
        self.assertTrue(result["logged"])
        saved = self.firebase.saved[f"repair_history/{result['log_id']}"]
        self.assertEqual(saved["session_id"], "firebase-session")
        self.assertEqual(saved["action_type"], "move")

    def test_get_agent_config_defaults(self):
        config = self.ops.get_agent_config()
        self.assertEqual(config["max_turns"], 10)
        self.assertEqual(config["llm_model"], "claude-sonnet-4-5")
        self.assertTrue(config["enabled"])
        self.assertTrue(config["fallback_to_random_repair"])

    def test_save_and_load_agent_config(self):
        payload = {
            "max_turns": 8,
            "llm_model": "claude-sonnet-4-5",
            "enabled": True,
            "fallback_to_random_repair": False,
        }
        success, config_id = self.firebase.save_agent_config("default", payload)
        self.assertTrue(success)
        loaded = self.ops.get_agent_config("default")
        self.assertEqual(loaded["max_turns"], 8)


if __name__ == "__main__":
    unittest.main()
