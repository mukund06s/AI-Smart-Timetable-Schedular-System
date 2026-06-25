"""
Phase 3 Day 18-19 metrics collection tests.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.metrics_collector import MetricsCollector
from agent.mock_repair_client import AutoRepairMockClient
from agent.scenarios import (
    build_research_scenario_a,
    build_research_scenario_b,
    build_research_scenario_c,
    build_test_case_1_single_faculty_clash,
)
from genetic_algorithm import GeneticAlgorithm
from tests.test_phase3_cases import MockClashDetector


class Phase3MetricsTests(unittest.TestCase):
    def setUp(self):
        self.collector = MetricsCollector(
            firebase_manager=None,
            genetic_algorithm=GeneticAlgorithm(),
            clash_detector_cls=MockClashDetector,
        )
        self.client_factory = lambda sched, cons: AutoRepairMockClient(
            sched, cons, max_moves=25
        )

    def test_legacy_repair_metrics(self):
        schedule, constraints, _ = build_test_case_1_single_faculty_clash()
        result = self.collector.run_legacy_repair(
            schedule, constraints, "Test_Legacy_SingleClash"
        )
        self.assertEqual(result.method, "legacy_intelligent_repair")
        self.assertEqual(result.clashes_at_start, 1)
        self.assertFalse(result.explainable)
        self.assertGreaterEqual(result.repair_time_seconds, 0)

    def test_agentic_repair_metrics(self):
        schedule, constraints, _ = build_test_case_1_single_faculty_clash()
        result = self.collector.run_agentic_repair(
            schedule,
            constraints,
            "Test_Agentic_SingleClash",
            llm_client_factory=self.client_factory,
        )
        self.assertEqual(result.method, "agentic_repair")
        self.assertTrue(result.explainable)
        self.assertGreaterEqual(result.clash_resolution_rate_pct, 0)

    def test_compare_legacy_vs_agentic(self):
        schedule, constraints, _ = build_test_case_1_single_faculty_clash()
        comparison = self.collector.compare_repair_methods(
            schedule,
            constraints,
            "Compare_SingleClash",
            llm_client_factory=self.client_factory,
        )
        self.assertEqual(comparison.legacy.clashes_at_start, comparison.agentic.clashes_at_start)
        self.assertTrue(comparison.agentic.explainable)
        self.assertFalse(comparison.legacy.explainable)

    def test_research_scenarios_a_b_c(self):
        results = self.collector.run_all_research_scenarios(
            llm_client_factory=self.client_factory
        )
        self.assertEqual(len(results), 3)
        names = {r.scenario_name for r in results}
        self.assertIn("Scenario_A_Sem2_BTECH_2Sections", names)
        self.assertIn("Scenario_B_Sem4_BTECH_2Sections", names)
        self.assertIn("Scenario_C_CrossSemester_Sem2_Sem4", names)

    def test_metrics_tables(self):
        schedule, constraints, _ = build_test_case_1_single_faculty_clash()
        comparison = self.collector.compare_repair_methods(
            schedule,
            constraints,
            "Table_Test",
            llm_client_factory=self.client_factory,
        )
        table1 = self.collector.build_comparison_table([comparison])
        table2 = self.collector.build_time_complexity_table([comparison])
        self.assertEqual(len(table1), 2)
        self.assertEqual(len(table2), 1)
        self.assertIn("Resolution Rate %", table1[0])
        self.assertIn("Agentic Turns", table2[0])

    def test_resolution_rate_calculation(self):
        self.assertEqual(MetricsCollector.compute_resolution_rate(10, 1), 90.0)
        self.assertEqual(MetricsCollector.compute_resolution_rate(0, 0), 100.0)


if __name__ == "__main__":
    unittest.main()
