"""
End-to-end tests for TimetableAgent ReAct loop with mocked Anthropic tool-calling.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.memory import AgentMemory
from agent.prompts import build_initial_user_message, build_system_prompt
from agent.timetable_agent import TimetableAgent
from agent.tools import ToolRegistry
from tests.agent_test_helpers import (
    MockAnthropicClient,
    MockFirebase,
    build_sample_schedule,
)


class TimetableAgentE2ETests(unittest.TestCase):
    def test_prompts_match_blueprint_structure(self):
        prompt = build_system_prompt({})
        self.assertIn("[ROLE]", prompt)
        self.assertIn("[CONTEXT]", prompt)
        self.assertIn("[CONSTRAINTS YOU MUST RESPECT]", prompt)
        self.assertIn("[TOOLS]", prompt)
        self.assertIn("[REASONING STYLE]", prompt)
        self.assertIn("[OUTPUT FORMAT]", prompt)
        self.assertIn("THOUGHT:", prompt)
        self.assertIn("ACTION:", prompt)

    def test_react_loop_fixes_injected_faculty_clash(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        tools = ToolRegistry(MockFirebase())
        tools.bind_context(schedule=schedule, constraints={}, session_id="e2e")
        initial = tools.execute("tool_read_clashes", {}, schedule)
        self.assertGreaterEqual(initial["clash_count"], 1)

        mock_steps = [
            {
                "name": "tool_move_class",
                "input": {
                    "school_key": "BTECH",
                    "batch_key": "Sem_2_Section_B",
                    "from_day": "Monday",
                    "from_slot": "09:00-10:00",
                    "to_day": "Tuesday",
                    "to_slot": "10:00-11:00",
                },
            },
            {
                "name": "tool_log_repair",
                "input": {
                    "action_type": "move",
                    "from_slot": {
                        "day": "Monday",
                        "slot": "09:00-10:00",
                        "batch_key": "Sem_2_Section_B",
                    },
                    "to_slot": {
                        "day": "Tuesday",
                        "slot": "10:00-11:00",
                        "batch_key": "Sem_2_Section_B",
                    },
                    "reason": "Moved clashing Chemistry class to free slot",
                    "result": True,
                    "clash_type": "Faculty Clash",
                    "faculty_or_room": "Dr. Mehta",
                },
            },
            None,
        ]

        agent = TimetableAgent(
            firebase_manager=MockFirebase(),
            llm_client=MockAnthropicClient(mock_steps),
        )
        repaired, summary = agent.repair_schedule(
            schedule=schedule,
            clashes=initial["clashes"],
            constraints={},
            timetable_key="BTECH_Sem2",
            program="BTECH",
            semester=2,
        )

        verify = tools.execute("tool_verify_schedule", {}, repaired)
        self.assertEqual(verify["clash_count"], 0)
        self.assertEqual(summary["status"], "completed")
        self.assertGreaterEqual(agent.client.messages.call_count, 1)

    def test_agent_memory_summary(self):
        memory = AgentMemory(
            schedule=build_sample_schedule(with_faculty_clash=False),
            clashes=[],
            timetable_key="BTECH_Sem2",
            program="BTECH",
            semester=2,
        )
        summary = memory.get_repair_summary()
        self.assertEqual(summary["clashes_found"], 0)
        self.assertEqual(summary["status"], "in_progress")

    def test_no_clash_schedule_completes_immediately(self):
        schedule = build_sample_schedule(with_faculty_clash=False)
        agent = TimetableAgent(
            firebase_manager=MockFirebase(),
            llm_client=MockAnthropicClient([]),
        )
        repaired, summary = agent.repair_schedule(
            schedule=schedule,
            clashes=[],
            constraints={},
        )
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["clashes_found"], 0)
        self.assertIs(repaired, schedule)

    def test_initial_user_message_format(self):
        clashes = [
            {
                "type": "Faculty Clash",
                "faculty": "Dr. Mehta",
                "time": "Monday at 09:00-10:00",
                "details": "Dr. Mehta assigned to 2 classes simultaneously",
            }
        ]
        message = build_initial_user_message(clashes)
        self.assertIn("Please repair the following 1 clashes", message)
        self.assertIn("Dr. Mehta", message)
        self.assertIn("Use your tools to fix each clash one by one.", message)

    def test_missing_llm_client_fails_without_api_key(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        tools = ToolRegistry(MockFirebase())
        initial = tools.execute("tool_read_clashes", {}, schedule)
        agent = TimetableAgent(firebase_manager=MockFirebase(), llm_client=None, api_key="")
        repaired, summary = agent.repair_schedule(
            schedule=schedule,
            clashes=initial["clashes"],
            constraints={},
        )
        self.assertEqual(summary["status"], "failed")
        verify = tools.execute("tool_verify_schedule", {}, repaired)
        self.assertGreater(verify["clash_count"], 0)


if __name__ == "__main__":
    unittest.main()
