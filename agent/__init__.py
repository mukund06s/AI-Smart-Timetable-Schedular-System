"""
Agentic AI layer for autonomous timetable clash repair.
"""

from agent.edge_cases import execute_tool_with_revert_guard, save_local_repair_backup
from agent.explain_repair import explain_repair_plain_english
from agent.firebase_ops import AgentFirebaseOps
from agent.integration import run_agentic_clash_repair, categorize_clashes, get_anthropic_api_key
from agent.memory import AgentMemory
from agent.metrics_collector import MetricsCollector, ComparisonResult, RepairMethodResult
from agent.prompts import build_initial_user_message, build_system_prompt
from agent.research_export import export_research_bundle, export_metrics_csv, generate_research_figures
from agent.scenarios import PHASE3_TEST_CASES, RESEARCH_SCENARIOS
from agent.timetable_agent import TimetableAgent
from agent.tools import ToolRegistry

__all__ = [
    "AgentFirebaseOps",
    "AgentMemory",
    "ToolRegistry",
    "TimetableAgent",
    "MetricsCollector",
    "ComparisonResult",
    "RepairMethodResult",
    "run_agentic_clash_repair",
    "categorize_clashes",
    "get_anthropic_api_key",
    "export_research_bundle",
    "export_metrics_csv",
    "generate_research_figures",
    "PHASE3_TEST_CASES",
    "RESEARCH_SCENARIOS",
    "build_initial_user_message",
    "build_system_prompt",
    "execute_tool_with_revert_guard",
    "explain_repair_plain_english",
]
