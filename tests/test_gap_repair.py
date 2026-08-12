"""
Tests for scheduling gap detection and deterministic gap repair.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gap_repair import deterministic_gap_repair
from agent.integration import run_agentic_clash_repair
from tests.agent_test_helpers import MockFirebase, build_sample_schedule
from utils.clash_analyzer import ClashAnalyzer


class MockClashDetector:
    def detect_all_clashes(
        self,
        schedule,
        existing_faculty_schedules=None,
        existing_room_schedules=None,
    ):
        return ClashAnalyzer().detect_all_clashes(
            schedule,
            existing_faculty_schedules=existing_faculty_schedules,
            existing_room_schedules=existing_room_schedules,
        )


class GapRepairTests(unittest.TestCase):
    def test_detect_scheduling_gaps_finds_missing_tutorial(self):
        schedule = build_sample_schedule()
        constraints = {
            "subjects": [
                {
                    "name": "Calculus Tutorial",
                    "section": "A",
                    "type": "Tutorial",
                    "weekly_hours": 1,
                    "faculty": "Dr. Singh",
                    "assigned_room": "LH-102",
                    "program": "BTECH",
                    "semester": "2",
                }
            ],
            "program": "BTECH",
            "semester": "2",
        }
        gaps = ClashAnalyzer().detect_scheduling_gaps(schedule, constraints)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["subject"], "Calculus Tutorial")
        self.assertEqual(gaps[0]["missing_hours"], 1)

    def test_deterministic_gap_repair_places_missing_class(self):
        schedule = build_sample_schedule()
        constraints = {
            "subjects": [
                {
                    "name": "Calculus Tutorial",
                    "section": "A",
                    "type": "Tutorial",
                    "weekly_hours": 1,
                    "faculty": "Dr. Singh",
                    "assigned_room": "LH-102",
                    "program": "BTECH",
                    "semester": "2",
                }
            ],
            "program": "BTECH",
            "semester": "2",
        }
        gaps = ClashAnalyzer().detect_scheduling_gaps(schedule, constraints)
        placed = deterministic_gap_repair(schedule, constraints, gaps)
        self.assertEqual(placed, 1)
        remaining = ClashAnalyzer().detect_scheduling_gaps(schedule, constraints)
        self.assertEqual(len(remaining), 0)

    def test_integration_runs_gap_fallback_without_llm(self):
        schedule = build_sample_schedule()
        constraints = {
            "subjects": [
                {
                    "name": "Calculus Tutorial",
                    "section": "A",
                    "type": "Tutorial",
                    "weekly_hours": 1,
                    "faculty": "Dr. Singh",
                    "assigned_room": "LH-102",
                    "program": "BTECH",
                    "semester": "2",
                }
            ],
            "program": "BTECH",
            "semester": "2",
        }
        gaps = ClashAnalyzer().detect_scheduling_gaps(schedule, constraints)
        repaired, summary, remaining, log = run_agentic_clash_repair(
            firebase_manager=MockFirebase(),
            genetic_algorithm=None,
            schedule=schedule,
            clashes=[],
            scheduling_gaps=gaps,
            constraints=constraints,
            program="BTECH",
            semester=2,
            clash_detector=MockClashDetector(),
            api_key="",
            enable_fallback=True,
        )
        self.assertTrue(summary.get("gap_fallback_used"))
        self.assertEqual(summary.get("remaining_gaps", 1), 0)
        self.assertEqual(len(remaining), 0)


if __name__ == "__main__":
    unittest.main()
