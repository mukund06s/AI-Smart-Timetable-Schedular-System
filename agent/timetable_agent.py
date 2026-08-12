"""
The core agentic AI class.
Wraps an LLM (Claude claude-sonnet-4-5) with a set of timetable tools.
Uses a ReAct-style (Reason + Act) loop for multi-turn repair.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.firebase_ops import AgentFirebaseOps
from agent.memory import AgentMemory
from agent.prompts import build_initial_user_message, build_system_prompt
from agent.edge_cases import call_llm_with_retry, execute_tool_with_revert_guard
from agent.tools import ToolRegistry
from config.settings import AGENT_SETTINGS
from utils.logging_config import get_logger, log_exception

logger = get_logger(__name__)


class TimetableAgent:
    """Autonomous timetable repair agent with Anthropic tool-calling."""

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        firebase_manager=None,
        llm_client=None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 10,
    ):
        self.firebase = firebase_manager
        self.firebase_ops = AgentFirebaseOps(firebase_manager)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.tools = ToolRegistry(firebase_manager)
        self.memory: Optional[AgentMemory] = None
        self.max_turns = max_turns
        self.model = model or self.DEFAULT_MODEL
        self.conversation_history: List[dict] = []
        self._constraints: dict = {}

        agent_config = self.firebase_ops.get_agent_config()
        if agent_config:
            self.max_turns = agent_config.get("max_turns", self.max_turns)
            self.model = agent_config.get("llm_model", self.model)

        if llm_client is not None:
            self.client = llm_client
        elif self.api_key:
            from agent.gemini_wrapper import GeminiAnthropicWrapper
            self.client = GeminiAnthropicWrapper(api_key=self.api_key)
        else:
            self.client = None


    def repair_schedule(
        self,
        schedule: dict,
        clashes: list,
        constraints: dict,
        scheduling_gaps: Optional[list] = None,
        on_turn_callback: Optional[Callable[[int, Any], None]] = None,
        timetable_key: str = "",
        program: str = "",
        semester: Optional[int] = None,
    ) -> Tuple[dict, dict]:
        """
        Main entry point.
        Given a schedule with clashes and/or incomplete subjects, returns a repaired schedule.
        Uses multi-turn LLM loop with tool-calling.
        """
        scheduling_gaps = scheduling_gaps or []
        self.memory = AgentMemory(
            schedule=schedule,
            clashes=clashes,
            scheduling_gaps=scheduling_gaps,
            timetable_key=timetable_key,
            program=program,
            semester=semester,
        )
        self.tools.bind_context(
            schedule=schedule,
            constraints=constraints,
            memory=self.memory,
            session_id=self.memory.session_id,
        )
        self._constraints = constraints

        if not clashes and not scheduling_gaps:
            self.memory.status = "completed"
            self.memory.ended_at = datetime.now(timezone.utc)
            self.memory.save_to_firebase(self.firebase)
            return schedule, self.memory.get_repair_summary()

        if self.client is None:
            self.memory.status = "failed"
            self.memory.ended_at = datetime.now(timezone.utc)
            self.memory.save_to_firebase(self.firebase)
            return schedule, self.memory.get_repair_summary()

        return self._run_agent_loop(
            schedule=schedule,
            clashes=clashes,
            scheduling_gaps=scheduling_gaps,
            constraints=constraints,
            on_turn_callback=on_turn_callback,
        )

    def _build_system_prompt(self, constraints: dict) -> str:
        """
        Returns the system prompt that tells the LLM:
        - What its role is (timetable repair agent)
        - What tools it has
        - What constraints to respect
        - How to reason about clashes
        """
        return build_system_prompt(constraints)

    def _schedule_is_clean(self, verify_result: dict) -> bool:
        gaps = self._execute_tool(
            "tool_read_scheduling_gaps", {}, self.memory.current_schedule
        )
        return (
            verify_result.get("clash_count", 0) == 0
            and gaps.get("gap_count", 0) == 0
        )

    def _run_agent_loop(
        self,
        schedule: dict,
        clashes: list,
        scheduling_gaps: list,
        constraints: dict,
        on_turn_callback: Optional[Callable[[int, Any], None]] = None,
    ) -> Tuple[dict, dict]:
        """
        The core ReAct loop:

        While issues remain and turns < max_turns:
          1. Send schedule + clashes/gaps to LLM
          2. LLM responds with THOUGHT + ACTION
          3. Execute the ACTION (tool call)
          4. Get OBSERVATION (result)
          5. Send OBSERVATION back to LLM
          6. Re-check clashes and scheduling gaps
          7. If clean: break + return fixed schedule
        """
        messages: List[dict] = [
            {
                "role": "user",
                "content": build_initial_user_message(clashes, scheduling_gaps),
            }
        ]
        self.conversation_history = list(messages)

        for turn in range(self.max_turns):
            remaining = self._execute_tool(
                "tool_verify_schedule", {}, self.memory.current_schedule
            )
            if self._schedule_is_clean(remaining):
                self.memory.status = "completed"
                break

            try:
                response = call_llm_with_retry(
                    self.client,
                    max_retries=AGENT_SETTINGS.llm_max_retries,
                    retry_delay=AGENT_SETTINGS.llm_retry_delay_seconds,
                    model=self.model,
                    max_tokens=2000,
                    system=self._build_system_prompt(constraints),
                    tools=self.tools.get_all_tools(),
                    messages=messages,
                )
            except Exception as exc:
                log_exception(logger, "LLM API call failed after retries", exc)
                self.memory.llm_error = str(exc)
                self.memory.status = "llm_failed"
                break

            if on_turn_callback:
                on_turn_callback(turn + 1, response)

            self.memory.log_llm_turn(turn + 1, response.content)
            messages.append({"role": "assistant", "content": response.content})
            self.conversation_history.append(
                {"role": "assistant", "content": response.content}
            )

            if response.stop_reason == "end_turn":
                break

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue

                result = self._execute_tool(
                    block.name,
                    block.input,
                    self.memory.current_schedule,
                )
                self.memory.log_action(block.name, block.input, result)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

            if not tool_results:
                break

            messages.append({"role": "user", "content": tool_results})
            self.conversation_history.append(
                {"role": "user", "content": tool_results}
            )

            remaining = self._execute_tool(
                "tool_verify_schedule", {}, self.memory.current_schedule
            )
            if self._schedule_is_clean(remaining):
                self.memory.status = "completed"
                break

        if self.memory.status == "in_progress":
            remaining = self._execute_tool(
                "tool_verify_schedule", {}, self.memory.current_schedule
            )
            if self._schedule_is_clean(remaining):
                self.memory.status = "completed"
            elif self.memory.turns_taken >= self.max_turns:
                self.memory.status = "max_turns_exceeded"
            elif self.memory.repairs_applied or self.memory.escalations:
                self.memory.status = "partial"
            else:
                self.memory.status = "failed"

        self.memory.ended_at = datetime.now(timezone.utc)
        self.memory.save_to_firebase(self.firebase)
        return self.memory.current_schedule, self.memory.get_repair_summary()

    def _execute_tool(
        self, tool_name: str, tool_args: dict, schedule: dict
    ) -> dict:
        """Routes tool calls to ToolRegistry with clash-revert guard."""
        result, revert_info = execute_tool_with_revert_guard(
            self.tools,
            tool_name,
            tool_args,
            schedule,
            self._constraints,
        )
        if revert_info and self.memory:
            self.memory.record_revert(revert_info)
        return result

    def _format_clashes(self, clashes: list) -> str:
        lines = []
        for index, clash in enumerate(clashes, 1):
            entity = clash.get("faculty") or clash.get("room") or "Unknown"
            lines.append(
                f"{index}. [{clash.get('type', 'Unknown')}] {entity} "
                f"at {clash.get('time', '')} — {clash.get('details', clash.get('description', ''))}"
            )
        return "\n".join(lines)
