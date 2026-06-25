"""
Tests for reliability, security, and documentation improvements.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.input_validation import validate_tool_args
from agent.rate_limiter import AgentRateLimiter
from agent.statistical_analysis import analyze_comparison_results
from agent.metrics_collector import ComparisonResult, RepairMethodResult
from config.settings import AGENT_SETTINGS, APP_SETTINGS
from utils.logging_config import configure_logging, get_logger


class ImprovementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_logging()

    def test_logging_configures(self):
        logger = get_logger("test.improvements")
        self.assertEqual(logger.name, "test.improvements")

    def test_validate_tool_move_class_requires_fields(self):
        error = validate_tool_args("tool_move_class", {"school_key": "BTECH"})
        self.assertIsNotNone(error)
        self.assertIn("Missing", error)

    def test_validate_tool_read_clashes_optional(self):
        self.assertIsNone(validate_tool_args("tool_read_clashes", {}))

    def test_rate_limiter_blocks_after_max(self):
        limiter = AgentRateLimiter(max_repairs_per_hour=2)
        self.assertTrue(limiter.allow("test-user")[0])
        self.assertTrue(limiter.allow("test-user")[0])
        allowed, message = limiter.allow("test-user")
        self.assertFalse(allowed)
        self.assertIn("rate limit", message.lower())

    def test_settings_defaults(self):
        self.assertGreaterEqual(AGENT_SETTINGS.max_turns, 1)
        self.assertGreaterEqual(APP_SETTINGS.ga_max_attempts, 100)

    def test_statistical_analysis_on_comparison(self):
        legacy = RepairMethodResult(
            method="legacy",
            scenario_name="Test",
            clashes_at_start=10,
            clashes_after_fix=2,
            clash_resolution_rate_pct=80.0,
            repair_time_seconds=45.0,
            iterations_or_turns=25,
            escalations=0,
            explainable=False,
        )
        agentic = RepairMethodResult(
            method="agentic",
            scenario_name="Test",
            clashes_at_start=10,
            clashes_after_fix=0,
            clash_resolution_rate_pct=100.0,
            repair_time_seconds=20.0,
            iterations_or_turns=6,
            escalations=0,
            explainable=True,
        )
        report = analyze_comparison_results(
            [ComparisonResult("Test", legacy, agentic)]
        )
        self.assertEqual(report["scenarios_analyzed"], 1)
        self.assertIn("repair_time_seconds", report)
        self.assertIn("interpretation", report)

    def test_readme_exists(self):
        readme = ROOT / "README.md"
        self.assertTrue(readme.exists())
        content = readme.read_text(encoding="utf-8")
        self.assertIn("streamlit run app.py", content)

    def test_secrets_example_exists(self):
        example = ROOT / ".streamlit" / "secrets.toml.example"
        self.assertTrue(example.exists())
        self.assertIn("ANTHROPIC_API_KEY", example.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
