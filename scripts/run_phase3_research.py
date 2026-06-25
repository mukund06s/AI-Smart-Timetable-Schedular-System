"""
Run Phase 3 research benchmarks and export all paper artifacts.

Usage:
    python scripts/run_phase3_research.py
"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.metrics_collector import MetricsCollector
from agent.mock_repair_client import AutoRepairMockClient
from agent.research_export import export_research_bundle
from agent.scenarios import RESEARCH_SCENARIOS
from genetic_algorithm import GeneticAlgorithm


class _Detector:
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


def main():
    collector = MetricsCollector(
        genetic_algorithm=GeneticAlgorithm(),
        clash_detector_cls=_Detector,
    )
    client_factory = lambda sched, cons: AutoRepairMockClient(sched, cons, max_moves=30)
    results = collector.run_all_research_scenarios(llm_client_factory=client_factory)

    conversation_logs = {}
    schedules = {}
    for key, builder in RESEARCH_SCENARIOS.items():
        before, constraints, _, scenario_name = builder()
        semester = 4 if "scenario_b" in key else 2
        comparison = next(r for r in results if r.scenario_name == scenario_name)
        after = copy.deepcopy(before)
        collector.run_agentic_repair(
            after,
            constraints,
            scenario_name,
            llm_client_factory=client_factory,
            semester=semester,
        )
        session_id = comparison.agentic.session_id or f"session_{key}"
        conversation_logs[session_id] = [
            {"scenario": scenario_name, "metrics": comparison.agentic.to_dict()}
        ]
        schedules[scenario_name] = {"before": before, "after": after}

    bundle = export_research_bundle(
        results,
        conversation_logs=conversation_logs,
        before_after_schedules=schedules,
    )
    print("Phase 3 research bundle exported:")
    print(bundle["manifest"])


if __name__ == "__main__":
    main()
