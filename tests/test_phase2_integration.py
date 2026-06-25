"""
Phase 2 integration tests — pipeline hook, fallback, reports, UI helpers.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.integration import categorize_clashes, run_agentic_clash_repair
from tests.agent_test_helpers import MockAnthropicClient, MockFirebase, build_sample_schedule


class MockClashDetector:
    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager

    def detect_all_clashes(
        self,
        schedule,
        existing_faculty_schedules=None,
        existing_room_schedules=None,
    ):
        from utils.clash_analyzer import ClashAnalyzer

        return ClashAnalyzer().detect_all_clashes(
            schedule,
            existing_faculty_schedules=existing_faculty_schedules,
            existing_room_schedules=existing_room_schedules,
        )


class MockGeneticAlgorithm:
    def _intelligent_repair(self, schedule, constraints):
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, value in list(schedule[school][batch][day].items()):
                        if (
                            isinstance(value, dict)
                            and value.get("faculty") == "Dr. Mehta"
                            and batch == "Sem_2_Section_B"
                            and day == "Monday"
                            and slot == "09:00-10:00"
                        ):
                            schedule[school][batch][day][slot] = None
                            schedule[school][batch][day]["10:00-11:00"] = value


class Phase2IntegrationTests(unittest.TestCase):
    def test_categorize_clashes(self):
        clashes = [
            {"type": "Faculty Clash"},
            {"type": "Room Clash"},
            {"type": "Cross-Semester Faculty Clash"},
        ]
        counts = categorize_clashes(clashes)
        self.assertEqual(counts["faculty"], 1)
        self.assertEqual(counts["room"], 1)
        self.assertEqual(counts["cross_semester"], 1)

    def test_run_agentic_clash_repair_with_mock_llm(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        detector = MockClashDetector()
        clashes = detector.detect_all_clashes(schedule)
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
            None,
        ]
        repaired, summary, remaining, log = run_agentic_clash_repair(
            firebase_manager=MockFirebase(),
            genetic_algorithm=MockGeneticAlgorithm(),
            schedule=schedule,
            clashes=clashes,
            constraints={},
            program="BTECH",
            semester=2,
            clash_detector=detector,
            llm_client=MockAnthropicClient(mock_steps),
            api_key="test-key",
        )
        self.assertEqual(len(remaining), 0)
        self.assertEqual(summary["status"], "completed")
        self.assertGreater(len(log), 0)

    def test_fallback_runs_when_agent_fails(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        detector = MockClashDetector()
        clashes = detector.detect_all_clashes(schedule)
        repaired, summary, remaining, log = run_agentic_clash_repair(
            firebase_manager=MockFirebase(),
            genetic_algorithm=MockGeneticAlgorithm(),
            schedule=schedule,
            clashes=clashes,
            constraints={},
            program="BTECH",
            semester=2,
            clash_detector=detector,
            api_key="",
        )
        self.assertTrue(summary.get("fallback_used"))
        self.assertEqual(len(remaining), 0)
        self.assertTrue(any("Fallback" in line for line in log))

    def test_report_generator_agent_history(self):
        firebase = MockFirebase()
        firebase.save_repair_history(
            "repair-1",
            {
                "repair_id": "repair-1",
                "session_id": "session-1",
                "action_type": "move",
                "clash_type": "Faculty Clash",
                "faculty_or_room": "Dr. Mehta",
                "from_slot": {"day": "Monday", "slot": "09:00-10:00"},
                "to_slot": {"day": "Tuesday", "slot": "10:00-11:00"},
                "reason": "Resolved clash",
                "success": True,
                "timestamp": "2026-06-25T00:00:00",
            },
        )

        class ReportGenStub:
            def __init__(self, fb):
                self.firebase = fb

            def generate_agent_repair_history_report(self):
                history = self.firebase.get_repair_history(limit=200)
                import pandas as pd

                rows = []
                for entry in history:
                    rows.append(
                        {
                            "Repair ID": entry.get("repair_id"),
                            "Session ID": entry.get("session_id"),
                            "Action Type": entry.get("action_type"),
                        }
                    )
                return pd.DataFrame(rows)

        report = ReportGenStub(firebase).generate_agent_repair_history_report()
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0]["Action Type"], "move")


if __name__ == "__main__":
    unittest.main()
