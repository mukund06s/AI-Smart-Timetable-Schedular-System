"""
Phase 4 edge-case tests: max turns, LLM retry, Firebase backup, revert guard.
"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.edge_cases import (
    call_llm_with_retry,
    execute_tool_with_revert_guard,
    save_local_session_backup,
)
from agent.integration import run_agentic_clash_repair
from agent.memory import AgentMemory
from agent.timetable_agent import TimetableAgent
from agent.tools import ToolRegistry
from tests.agent_test_helpers import (
    MockAnthropicClient,
    MockAnthropicResponse,
    MockFirebase,
    MockTextBlock,
    MockToolUseBlock,
    build_sample_schedule,
)


class FailingFirebase(MockFirebase):
    def save_agent_session(self, session_id, session_data):
        return False, "simulated firebase failure"

    def save_repair_history(self, repair_id, repair_data):
        return False, "simulated firebase failure"


class FlakyLLMClient:
    """Fails first two calls, succeeds on third."""

    def __init__(self, success_response):
        self.failures = 0
        self.success_response = success_response
        self.messages = self

    def create(self, **kwargs):
        if self.failures < 2:
            self.failures += 1
            raise ConnectionError("transient API failure")
        return self.success_response


class NeverFixLLMClient:
    """Always returns a non-fixing tool call."""

    def __init__(self):
        self.messages = self
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return MockAnthropicResponse(
            [
                MockTextBlock("Analyzing clash."),
                MockToolUseBlock(
                    "tool_1",
                    "tool_read_clashes",
                    {},
                ),
            ],
            "tool_use",
        )


class Phase4EdgeCaseTests(unittest.TestCase):
    def test_llm_retry_succeeds_after_transient_failures(self):
        response = MockAnthropicResponse([MockTextBlock("ok")], "end_turn")
        client = FlakyLLMClient(response)
        result = call_llm_with_retry(client, max_retries=3, model="test")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(client.failures, 2)

    def test_llm_retry_raises_after_exhausted_retries(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("persistent failure")
        with self.assertRaises(RuntimeError):
            call_llm_with_retry(client, max_retries=2, model="test")
        self.assertEqual(client.messages.create.call_count, 2)

    def test_firebase_failure_saves_local_session_backup(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        tools = ToolRegistry(FailingFirebase())
        tools.bind_context(schedule=schedule, constraints={}, session_id="local")
        initial = tools.execute("tool_read_clashes", {}, schedule)

        agent = TimetableAgent(
            firebase_manager=FailingFirebase(),
            llm_client=MockAnthropicClient(
                [
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
            ),
            max_turns=3,
        )
        _, summary = agent.repair_schedule(
            schedule=schedule,
            clashes=initial["clashes"],
            constraints={},
        )
        self.assertFalse(summary.get("firebase_saved", True))
        self.assertTrue(summary.get("local_backup_path"))
        self.assertTrue(Path(summary["local_backup_path"]).exists())

    def test_revert_guard_restores_schedule_when_clashes_increase(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        snapshot = copy.deepcopy(schedule)
        tools = ToolRegistry(MockFirebase())
        tools.bind_context(schedule=schedule, constraints={}, session_id="revert")

        with patch("agent.edge_cases.clash_count") as mock_count:
            mock_count.side_effect = [1, 3]
            result, revert_info = execute_tool_with_revert_guard(
                tools,
                "tool_move_class",
                {
                    "school_key": "BTECH",
                    "batch_key": "Sem_2_Section_B",
                    "from_day": "Monday",
                    "from_slot": "09:00-10:00",
                    "to_day": "Tuesday",
                    "to_slot": "10:00-11:00",
                },
                schedule,
                {},
            )

        self.assertTrue(result.get("reverted"))
        self.assertIsNotNone(revert_info)
        self.assertEqual(schedule, snapshot)

    def test_max_turns_exceeded_status(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        analyzer = MagicMock()
        analyzer.detect_all_clashes.return_value = [{"type": "Faculty Clash"}]

        _, summary, _, _ = run_agentic_clash_repair(
            firebase_manager=MockFirebase(),
            genetic_algorithm=MagicMock(),
            schedule=copy.deepcopy(schedule),
            clashes=[{"type": "Faculty Clash"}],
            constraints={},
            program="BTECH",
            semester=2,
            clash_detector=analyzer,
            llm_client=NeverFixLLMClient(),
            enable_fallback=False,
        )
        self.assertEqual(summary.get("status"), "max_turns_exceeded")

    def test_max_turns_exceeded_triggers_fallback(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        analyzer = MagicMock()
        analyzer.detect_all_clashes.return_value = [{"type": "Faculty Clash"}]

        ga = MagicMock()
        ga._intelligent_repair.side_effect = lambda sched, cons: sched.update({})

        def _detect(sched, **kwargs):
            if ga._intelligent_repair.call_count >= 1:
                return []
            return [{"type": "Faculty Clash"}]

        analyzer.detect_all_clashes.side_effect = _detect

        _, summary, _, log = run_agentic_clash_repair(
            firebase_manager=MockFirebase(),
            genetic_algorithm=ga,
            schedule=copy.deepcopy(schedule),
            clashes=[{"type": "Faculty Clash"}],
            constraints={},
            program="BTECH",
            semester=2,
            clash_detector=analyzer,
            llm_client=NeverFixLLMClient(),
        )

        self.assertTrue(summary.get("fallback_used"))
        self.assertEqual(
            summary.get("agent_status_before_fallback"), "max_turns_exceeded"
        )
        self.assertTrue(any("max turns" in line.lower() for line in log))
        self.assertTrue(any("Fallback" in line for line in log))

    def test_llm_failed_status(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        analyzer = MagicMock()
        analyzer.detect_all_clashes.return_value = [{"type": "Faculty Clash"}]

        class AlwaysFailClient:
            messages = property(lambda self: self)

            def create(self, **kwargs):
                raise ConnectionError("API down")

        _, summary, _, log = run_agentic_clash_repair(
            firebase_manager=MockFirebase(),
            genetic_algorithm=MagicMock(),
            schedule=copy.deepcopy(schedule),
            clashes=[{"type": "Faculty Clash"}],
            constraints={},
            program="BTECH",
            semester=2,
            clash_detector=analyzer,
            llm_client=AlwaysFailClient(),
            enable_fallback=False,
        )
        self.assertEqual(summary.get("status"), "llm_failed")
        self.assertTrue(any("LLM API failed" in line for line in log))

    def test_llm_failed_triggers_fallback(self):
        schedule = build_sample_schedule(with_faculty_clash=True)

        class AlwaysFailClient:
            messages = property(lambda self: self)

            def create(self, **kwargs):
                raise ConnectionError("API down")

        analyzer = MagicMock()
        analyzer.detect_all_clashes.return_value = [{"type": "Faculty Clash"}]
        ga = MagicMock()

        _, summary, _, log = run_agentic_clash_repair(
            firebase_manager=MockFirebase(),
            genetic_algorithm=ga,
            schedule=copy.deepcopy(schedule),
            clashes=[{"type": "Faculty Clash"}],
            constraints={},
            program="BTECH",
            semester=2,
            clash_detector=analyzer,
            llm_client=AlwaysFailClient(),
        )
        self.assertTrue(summary.get("fallback_used"))
        self.assertEqual(summary.get("agent_status_before_fallback"), "llm_failed")
        self.assertTrue(any("LLM API failed" in line for line in log))

    def test_agent_records_reverted_repairs(self):
        schedule = build_sample_schedule(with_faculty_clash=True)
        memory = AgentMemory(schedule=schedule, clashes=[{"type": "Faculty Clash"}])
        memory.record_revert({"tool": "tool_move_class", "clashes_before": 1, "clashes_after": 2})
        summary = memory.get_repair_summary()
        self.assertEqual(len(summary["reverted_repairs"]), 1)

    def test_local_session_backup_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "session.json"
            with patch("agent.edge_cases.LOCAL_SESSION_DIR", Path(tmp)):
                path = save_local_session_backup({"session_id": "abc"}, "abc")
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
