"""
Phase 3 metrics collection — compare legacy repair vs agentic repair for research paper.
"""

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agent.integration import run_agentic_clash_repair
from agent.scenarios import RESEARCH_SCENARIOS, detect_scenario_clashes
from utils.clash_analyzer import ClashAnalyzer


@dataclass
class RepairMethodResult:
    method: str
    scenario_name: str
    clashes_at_start: int
    clashes_after_fix: int
    clash_resolution_rate_pct: float
    repair_time_seconds: float
    iterations_or_turns: int
    escalations: int
    explainable: bool
    fallback_used: bool = False
    tool_call_counts: Dict[str, int] = field(default_factory=dict)
    status: str = ""
    session_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonResult:
    scenario_name: str
    legacy: RepairMethodResult
    agentic: RepairMethodResult

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "legacy": self.legacy.to_dict(),
            "agentic": self.agentic.to_dict(),
        }


class MetricsCollector:
    """Collect and compare repair metrics for Phase 3 research experiments."""

    def __init__(self, firebase_manager=None, genetic_algorithm=None, clash_detector_cls=None):
        self.firebase = firebase_manager
        self.genetic_algorithm = genetic_algorithm
        self.clash_detector_cls = clash_detector_cls
        self.analyzer = ClashAnalyzer()

    def _make_clash_detector(self):
        if self.clash_detector_cls:
            return self.clash_detector_cls(self.firebase)
        return _SimpleClashDetector(self.analyzer)

    def run_legacy_repair(
        self,
        schedule: dict,
        constraints: dict,
        scenario_name: str,
        max_rounds: int = 30,
    ) -> RepairMethodResult:
        schedule_copy = copy.deepcopy(schedule)
        detector = self._make_clash_detector()
        existing_faculty = constraints.get("existing_faculty_schedules")
        existing_room = constraints.get("existing_room_schedules")

        start_clashes = detector.detect_all_clashes(
            schedule_copy,
            existing_faculty_schedules=existing_faculty,
            existing_room_schedules=existing_room,
        )
        start_count = len(start_clashes)
        start_time = time.perf_counter()
        iterations = 0
        remaining = start_clashes

        if self.genetic_algorithm and start_count:
            for _ in range(max_rounds):
                self.genetic_algorithm._intelligent_repair(schedule_copy, constraints)
                iterations += 1
                remaining = detector.detect_all_clashes(
                    schedule_copy,
                    existing_faculty_schedules=existing_faculty,
                    existing_room_schedules=existing_room,
                )
                if not remaining:
                    break

        elapsed = round(time.perf_counter() - start_time, 3)
        end_count = len(remaining)
        resolution = self.compute_resolution_rate(start_count, end_count)
        return RepairMethodResult(
            method="legacy_intelligent_repair",
            scenario_name=scenario_name,
            clashes_at_start=start_count,
            clashes_after_fix=end_count,
            clash_resolution_rate_pct=resolution,
            repair_time_seconds=elapsed,
            iterations_or_turns=iterations,
            escalations=0,
            explainable=False,
            fallback_used=False,
            status="completed" if end_count == 0 else "partial",
        )

    def run_agentic_repair(
        self,
        schedule: dict,
        constraints: dict,
        scenario_name: str,
        llm_client=None,
        llm_client_factory=None,
        enable_fallback: bool = False,
        program: str = "BTECH",
        semester: int = 2,
    ) -> RepairMethodResult:
        schedule_copy = copy.deepcopy(schedule)
        detector = self._make_clash_detector()
        existing_faculty = constraints.get("existing_faculty_schedules")
        existing_room = constraints.get("existing_room_schedules")

        start_clashes = detector.detect_all_clashes(
            schedule_copy,
            existing_faculty_schedules=existing_faculty,
            existing_room_schedules=existing_room,
        )
        start_count = len(start_clashes)
        start_time = time.perf_counter()

        client = llm_client
        if client is None and llm_client_factory:
            client = llm_client_factory(schedule_copy, constraints)

        firebase = _MetricsFirebase(enable_fallback=enable_fallback)
        _, summary, remaining, _ = run_agentic_clash_repair(
            firebase_manager=firebase,
            genetic_algorithm=self.genetic_algorithm,
            schedule=schedule_copy,
            clashes=start_clashes,
            constraints=constraints,
            program=program,
            semester=semester,
            clash_detector=detector,
            llm_client=client,
            api_key="phase3-test-key" if client else "",
            enable_fallback=enable_fallback,
        )

        elapsed = round(time.perf_counter() - start_time, 3)
        end_count = len(remaining)
        resolution = self.compute_resolution_rate(start_count, end_count)
        tool_counts = _extract_tool_counts(summary)

        return RepairMethodResult(
            method="agentic_repair",
            scenario_name=scenario_name,
            clashes_at_start=start_count,
            clashes_after_fix=end_count,
            clash_resolution_rate_pct=resolution,
            repair_time_seconds=elapsed,
            iterations_or_turns=summary.get("turns_used", 0),
            escalations=summary.get("escalated", 0),
            explainable=True,
            fallback_used=bool(summary.get("fallback_used")),
            tool_call_counts=tool_counts,
            status=summary.get("status", ""),
            session_id=summary.get("session_id", ""),
        )

    def compare_repair_methods(
        self,
        schedule: dict,
        constraints: dict,
        scenario_name: str,
        llm_client=None,
        llm_client_factory=None,
        program: str = "BTECH",
        semester: int = 2,
    ) -> ComparisonResult:
        legacy = self.run_legacy_repair(schedule, constraints, scenario_name)
        agentic = self.run_agentic_repair(
            schedule,
            constraints,
            scenario_name,
            llm_client=llm_client,
            llm_client_factory=llm_client_factory,
            enable_fallback=False,
            program=program,
            semester=semester,
        )
        return ComparisonResult(
            scenario_name=scenario_name,
            legacy=legacy,
            agentic=agentic,
        )

    def run_all_research_scenarios(
        self,
        llm_client_factory=None,
    ) -> List[ComparisonResult]:
        results: List[ComparisonResult] = []
        for key, builder in RESEARCH_SCENARIOS.items():
            schedule, constraints, _, scenario_name = builder()
            semester = 4 if "Sem4" in scenario_name or "scenario_b" in key else 2
            results.append(
                self.compare_repair_methods(
                    schedule,
                    constraints,
                    scenario_name,
                    llm_client_factory=llm_client_factory,
                    semester=semester,
                )
            )
        return results

    @staticmethod
    def compute_resolution_rate(start_count: int, end_count: int) -> float:
        if start_count <= 0:
            return 100.0
        resolved = max(start_count - end_count, 0)
        return round((resolved / start_count) * 100, 2)

    @staticmethod
    def build_comparison_table(results: List[ComparisonResult]) -> List[dict]:
        rows = []
        for result in results:
            rows.append(
                {
                    "Scenario": result.scenario_name,
                    "Method": "Legacy Intelligent Repair",
                    "Clashes At Start": result.legacy.clashes_at_start,
                    "Clashes After Fix": result.legacy.clashes_after_fix,
                    "Resolution Rate %": result.legacy.clash_resolution_rate_pct,
                    "Repair Time (sec)": result.legacy.repair_time_seconds,
                    "Turns/Iterations": result.legacy.iterations_or_turns,
                    "Escalations": result.legacy.escalations,
                    "Explainable": result.legacy.explainable,
                }
            )
            rows.append(
                {
                    "Scenario": result.scenario_name,
                    "Method": "Agentic Repair",
                    "Clashes At Start": result.agentic.clashes_at_start,
                    "Clashes After Fix": result.agentic.clashes_after_fix,
                    "Resolution Rate %": result.agentic.clash_resolution_rate_pct,
                    "Repair Time (sec)": result.agentic.repair_time_seconds,
                    "Turns/Iterations": result.agentic.iterations_or_turns,
                    "Escalations": result.agentic.escalations,
                    "Explainable": result.agentic.explainable,
                }
            )
        return rows

    @staticmethod
    def build_time_complexity_table(results: List[ComparisonResult]) -> List[dict]:
        rows = []
        for result in results:
            rows.append(
                {
                    "Scenario": result.scenario_name,
                    "Legacy Time (sec)": result.legacy.repair_time_seconds,
                    "Agentic Time (sec)": result.agentic.repair_time_seconds,
                    "Legacy Iterations": result.legacy.iterations_or_turns,
                    "Agentic Turns": result.agentic.iterations_or_turns,
                    "Time Improvement (sec)": round(
                        result.legacy.repair_time_seconds - result.agentic.repair_time_seconds,
                        3,
                    ),
                }
            )
        return rows


class _SimpleClashDetector:
    def __init__(self, analyzer: ClashAnalyzer):
        self.analyzer = analyzer

    def detect_all_clashes(
        self,
        schedule,
        existing_faculty_schedules=None,
        existing_room_schedules=None,
    ):
        return self.analyzer.detect_all_clashes(
            schedule,
            existing_faculty_schedules=existing_faculty_schedules,
            existing_room_schedules=existing_room_schedules,
        )


class _MetricsFirebase:
    def __init__(self, enable_fallback: bool = True):
        self._config = {
            "max_turns": 10,
            "llm_model": "claude-sonnet-4-5",
            "enabled": True,
            "fallback_to_random_repair": enable_fallback,
        }

    def get_agent_config(self, config_id: str = "default"):
        return dict(self._config)

    def save_agent_session(self, session_id, session_data):
        return True, session_id

    def save_repair_history(self, repair_id, repair_data):
        return True, repair_id


def _extract_tool_counts(summary: dict) -> Dict[str, int]:
    if summary.get("tool_call_counts"):
        return dict(summary["tool_call_counts"])
    counts: Dict[str, int] = {}
    repairs = summary.get("repairs_applied", [])
    for repair in repairs:
        action = repair.get("action_type", "unknown")
        counts[action] = counts.get(action, 0) + 1
    if summary.get("escalated"):
        counts["tool_escalate"] = summary.get("escalated", 0)
    return counts
