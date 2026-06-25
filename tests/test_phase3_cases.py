"""
Phase 3 Day 15-17 test cases from IMPLEMENTATION_BLUEPRINT_AND_PRD.md
"""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.integration import run_agentic_clash_repair
from agent.metrics_collector import MetricsCollector
from agent.mock_repair_client import AutoRepairMockClient
from agent.scenarios import PHASE3_TEST_CASES
from genetic_algorithm import GeneticAlgorithm
from tests.agent_test_helpers import MockFirebase


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


class Phase3CaseTests(unittest.TestCase):
    def setUp(self):
        self.detector = MockClashDetector()
        self.ga = GeneticAlgorithm()
        self.firebase = MockFirebase()

    def _run_case(self, case_name: str):
        schedule, constraints, clashes = PHASE3_TEST_CASES[case_name]()
        schedule_copy = copy.deepcopy(schedule)
        client = AutoRepairMockClient(schedule_copy, constraints, max_moves=30)
        repaired, summary, remaining, _ = run_agentic_clash_repair(
            firebase_manager=self.firebase,
            genetic_algorithm=self.ga,
            schedule=schedule_copy,
            clashes=clashes,
            constraints=constraints,
            program="BTECH",
            semester=2,
            clash_detector=self.detector,
            llm_client=client,
            enable_fallback=False,
        )
        return repaired, summary, remaining, clashes

    def test_case_1_single_faculty_clash(self):
        _, summary, remaining, clashes = self._run_case("case_1_single_faculty_clash")
        self.assertEqual(len(clashes), 1)
        self.assertEqual(len(remaining), 0)
        self.assertEqual(summary["status"], "completed")

    def test_case_2_multiple_faculty_clashes(self):
        _, summary, remaining, clashes = self._run_case("case_2_multiple_faculty_clashes")
        self.assertGreaterEqual(len(clashes), 2)
        self.assertEqual(len(remaining), 0)
        self.assertEqual(summary["status"], "completed")

    def test_case_3_room_clash(self):
        schedule, constraints, clashes = PHASE3_TEST_CASES["case_3_room_clash"]()
        self.assertEqual(len(clashes), 1)
        self.assertEqual(clashes[0]["type"], "Room Clash")
        schedule_copy = copy.deepcopy(schedule)
        client = AutoRepairMockClient(schedule_copy, constraints, max_moves=10)
        _, summary, remaining, _ = run_agentic_clash_repair(
            firebase_manager=self.firebase,
            genetic_algorithm=self.ga,
            schedule=schedule_copy,
            clashes=clashes,
            constraints=constraints,
            program="BTECH",
            semester=2,
            clash_detector=self.detector,
            llm_client=client,
            enable_fallback=False,
        )
        self.assertEqual(len(remaining), 0)
        self.assertEqual(summary["status"], "completed")

    def test_case_4_cross_semester_clash(self):
        schedule, constraints, clashes = PHASE3_TEST_CASES["case_4_cross_semester_clash"]()
        self.assertTrue(any("Cross-Semester" in c["type"] for c in clashes))
        schedule_copy = copy.deepcopy(schedule)
        client = AutoRepairMockClient(schedule_copy, constraints, max_moves=10)
        _, summary, remaining, _ = run_agentic_clash_repair(
            firebase_manager=self.firebase,
            genetic_algorithm=self.ga,
            schedule=schedule_copy,
            clashes=clashes,
            constraints=constraints,
            program="BTECH",
            semester=2,
            clash_detector=self.detector,
            llm_client=client,
            enable_fallback=True,
        )
        self.assertIn(summary["status"], {"completed", "partial"})

    def test_case_5_unsolvable_escalation(self):
        schedule, constraints, clashes = PHASE3_TEST_CASES["case_5_unsolvable_clash"]()
        self.assertGreaterEqual(len(clashes), 1)
        schedule_copy = copy.deepcopy(schedule)
        client = AutoRepairMockClient(schedule_copy, constraints, max_moves=5)
        _, summary, remaining, _ = run_agentic_clash_repair(
            firebase_manager=self.firebase,
            genetic_algorithm=self.ga,
            schedule=schedule_copy,
            clashes=clashes,
            constraints=constraints,
            program="BTECH",
            semester=2,
            clash_detector=self.detector,
            llm_client=client,
            enable_fallback=False,
        )
        self.assertGreater(summary.get("escalated", 0), 0)
        self.assertGreater(len(remaining), 0)

    def test_case_6_stress_10_plus_clashes(self):
        schedule, constraints, clashes = PHASE3_TEST_CASES["case_6_stress_clashes"]()
        self.assertGreaterEqual(len(clashes), 10)
        schedule_copy = copy.deepcopy(schedule)
        client = AutoRepairMockClient(schedule_copy, constraints, max_moves=40)
        _, summary, remaining, _ = run_agentic_clash_repair(
            firebase_manager=self.firebase,
            genetic_algorithm=self.ga,
            schedule=schedule_copy,
            clashes=clashes,
            constraints=constraints,
            program="BTECH",
            semester=2,
            clash_detector=self.detector,
            llm_client=client,
            enable_fallback=True,
        )
        self.assertIn(summary["status"], {"completed", "partial"})
        self.assertLess(len(remaining), len(clashes))


if __name__ == "__main__":
    unittest.main()
