"""
Phase 3 Day 20-21 research export tests.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.metrics_collector import MetricsCollector
from agent.mock_repair_client import AutoRepairMockClient
from agent.research_export import (
    export_before_after_schedules,
    export_conversation_logs,
    export_metrics_csv,
    export_research_bundle,
    generate_research_figures,
)
from agent.scenarios import build_test_case_1_single_faculty_clash
from genetic_algorithm import GeneticAlgorithm
from tests.test_phase3_cases import MockClashDetector


class Phase3ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.collector = MetricsCollector(
            genetic_algorithm=GeneticAlgorithm(),
            clash_detector_cls=MockClashDetector,
        )
        self.client_factory = lambda sched, cons: AutoRepairMockClient(
            sched, cons, max_moves=15
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_conversation_logs(self):
        logs = [{"turn": 1, "tool": "tool_move_class", "result": {"success": True}}]
        path = export_conversation_logs(logs, "session-test-1", self.temp_dir)
        self.assertTrue(Path(path).exists())
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(loaded[0]["tool"], "tool_move_class")

    def test_export_before_after_schedules(self):
        before = {"BTECH": {"Sem_2_Section_A": {}}}
        after = {"BTECH": {"Sem_2_Section_A": {"Monday": {}}}}
        paths = export_before_after_schedules(
            before, after, "Export_Test", self.temp_dir
        )
        self.assertTrue(Path(paths["before"]).exists())
        self.assertTrue(Path(paths["after"]).exists())

    def test_export_metrics_csv(self):
        schedule, constraints, _ = build_test_case_1_single_faculty_clash()
        comparison = self.collector.compare_repair_methods(
            schedule,
            constraints,
            "CSV_Test",
            llm_client_factory=self.client_factory,
        )
        paths = export_metrics_csv([comparison], self.temp_dir)
        self.assertTrue(Path(paths["table1_clash_resolution_comparison"]).exists())
        self.assertTrue(Path(paths["table2_time_complexity"]).exists())
        self.assertTrue(Path(paths["research_metrics_combined"]).exists())

    def test_generate_research_figures(self):
        schedule, constraints, _ = build_test_case_1_single_faculty_clash()
        comparison = self.collector.compare_repair_methods(
            schedule,
            constraints,
            "Figure_Test",
            llm_client_factory=self.client_factory,
        )
        figures = generate_research_figures([comparison], self.temp_dir)
        self.assertIn("figure1_clash_comparison", figures)
        self.assertIn("figure2_turns_distribution", figures)
        self.assertIn("figure3_tool_frequency", figures)
        self.assertTrue(Path(figures["figure1_clash_comparison"]).exists())

    def test_export_research_bundle(self):
        schedule, constraints, _ = build_test_case_1_single_faculty_clash()
        comparison = self.collector.compare_repair_methods(
            schedule,
            constraints,
            "Bundle_Test",
            llm_client_factory=self.client_factory,
        )
        bundle = export_research_bundle(
            [comparison],
            conversation_logs={
                comparison.agentic.session_id or "session-bundle": [
                    {"turn": 1, "tool": "tool_verify_schedule"}
                ]
            },
            before_after_schedules={
                "Bundle_Test": {"before": schedule, "after": schedule}
            },
            export_dir=self.temp_dir,
        )
        self.assertIn("metrics_csv", bundle)
        self.assertIn("figures", bundle)
        self.assertTrue(Path(bundle["manifest"]).exists())


if __name__ == "__main__":
    unittest.main()
