"""
Streamlit UI for the Agentic AI clash repair tab (Phase 4 polish).
"""

import copy
import queue
import threading
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from agent.explain_repair import explain_repair_plain_english
from agent.integration import (
    categorize_clashes,
    format_turn_log_entry,
    get_anthropic_api_key,
    run_agentic_clash_repair,
)
from utils.clash_analyzer import ClashAnalyzer


def _resolve_schedule_and_key():
    program = st.session_state.get("selected_program")
    semester = st.session_state.get("selected_semester")
    timetable_key = f"{program}_Sem{semester}" if program and semester else None
    schedule = None
    generated = st.session_state.get("generated_schedules", {})
    if timetable_key and timetable_key in generated:
        schedule = generated[timetable_key]
    return program, semester, timetable_key, schedule


def _build_constraints(firebase_manager, timetable_key: Optional[str]) -> dict:
    constraints = {"subjects": st.session_state.get("subjects_data", [])}
    if firebase_manager:
        constraints["existing_faculty_schedules"] = (
            firebase_manager.get_all_faculty_schedules(
                exclude_timetable_key=timetable_key
            )
        )
        constraints["existing_room_schedules"] = (
            firebase_manager.get_all_room_schedules(
                exclude_timetable_key=timetable_key
            )
        )
    return constraints


def _detect_clashes(schedule: dict, constraints: dict) -> list:
    analyzer = ClashAnalyzer()
    return analyzer.detect_all_clashes(
        schedule,
        existing_faculty_schedules=constraints.get("existing_faculty_schedules"),
        existing_room_schedules=constraints.get("existing_room_schedules"),
    )


def render_repair_summary(summary: dict) -> None:
    st.markdown("#### REPAIR SUMMARY:")
    resolved = summary.get("clashes_fixed", 0)
    found = summary.get("clashes_found", 0)
    escalated = summary.get("escalated", 0)
    elapsed = summary.get("elapsed_seconds")
    turns = summary.get("turns_used", 0)
    session_id = summary.get("session_id", "N/A")
    max_turns = summary.get("max_turns", 10)

    if summary.get("status") == "completed" and found <= resolved:
        st.success(f"✅ {resolved}/{found or resolved} clashes resolved automatically")
    elif summary.get("status") == "max_turns_exceeded":
        st.warning(
            f"⚠️ Agent reached max turns ({turns}/{max_turns}). "
            f"Resolved {resolved}/{found} before fallback."
        )
    elif summary.get("status") == "llm_failed":
        st.error(f"❌ LLM API failed: {summary.get('llm_error', 'unknown error')}")
    else:
        st.warning(f"✅ {resolved}/{found} clashes resolved automatically")

    if escalated:
        st.warning(f"⚠️  {escalated} clash(es) escalated (no free slot found)")
    if summary.get("reverted_repairs"):
        st.info(
            f"↩️ {len(summary['reverted_repairs'])} repair(s) were reverted "
            "(would have introduced new clashes)."
        )
    if elapsed is not None:
        st.info(f"⏱️  Time taken: {elapsed} seconds")
    st.info(f"🔄  LLM turns used: {turns}/{max_turns}")
    if summary.get("firebase_saved", True):
        st.info(f"💾  Session saved to Firebase: {session_id}")
    elif summary.get("local_backup_path"):
        st.warning(
            f"💾  Firebase unavailable — local backup: {summary['local_backup_path']}"
        )
    else:
        st.info(f"💾  Session ID: {session_id}")
    if summary.get("fallback_used"):
        st.warning("⚠️ Legacy _intelligent_repair fallback was used after agent repair.")


def render_explain_repair_section(summary: dict) -> None:
    repairs = summary.get("repairs_applied") or []
    if not repairs:
        return

    st.markdown("#### 📝 Explain This Repair")
    for index, repair in enumerate(repairs):
        label = (
            f"{repair.get('action_type', 'repair')} — "
            f"{repair.get('from_slot', {}).get('day', '?')} "
            f"{repair.get('from_slot', {}).get('slot', '?')}"
        )
        if st.button(f"Explain repair #{index + 1}: {label}", key=f"explain_{index}"):
            with st.spinner("Generating plain-English explanation..."):
                explanation = explain_repair_plain_english(
                    repair,
                    api_key=get_anthropic_api_key() or None,
                )
            st.info(explanation)


def render_repair_history_dashboard(firebase_manager) -> None:
    st.markdown("#### 📊 Repair History Dashboard")
    if not firebase_manager:
        st.info("Firebase manager not available — dashboard requires Firebase.")
        return

    sessions = firebase_manager.get_agent_sessions(limit=50) or []
    history = firebase_manager.get_repair_history(limit=100) or []

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    completed = sum(1 for s in sessions if s.get("status") == "completed")
    partial = sum(1 for s in sessions if s.get("status") in ("partial", "max_turns_exceeded"))
    failed = sum(1 for s in sessions if s.get("status") in ("failed", "llm_failed"))
    total_fixed = sum(s.get("clashes_fixed", 0) for s in sessions)

    metric_col1.metric("Total Sessions", len(sessions))
    metric_col2.metric("Completed", completed)
    metric_col3.metric("Partial / Max Turns", partial)
    metric_col4.metric("Clashes Fixed", total_fixed)

    status_filter = st.selectbox(
        "Filter sessions by status",
        ["All", "completed", "partial", "max_turns_exceeded", "failed", "llm_failed"],
        key="agent_dashboard_status_filter",
    )

    if sessions:
        session_rows = []
        for session in sessions:
            status = session.get("status", "")
            if status_filter != "All" and status != status_filter:
                continue
            session_rows.append(
                {
                    "Session ID": session.get("session_id", session.get("id", "")),
                    "Timetable": session.get("timetable_key", ""),
                    "Status": status,
                    "Clashes Found": session.get("clashes_found", 0),
                    "Clashes Fixed": session.get("clashes_fixed", 0),
                    "Turns Used": session.get("turns_used", 0),
                }
            )
        if session_rows:
            st.markdown("##### Agent Sessions")
            st.dataframe(
                pd.DataFrame(session_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(f"No sessions with status '{status_filter}'.")

    if history:
        st.markdown("##### Repair Actions")
        st.dataframe(
            pd.DataFrame(history),
            use_container_width=True,
            hide_index=True,
            height=280,
        )
    elif not sessions:
        st.info("No repair history entries found in Firebase yet.")


def _run_repair_with_streaming(
    firebase_manager,
    clash_detector_cls,
    genetic_algorithm,
    schedule: dict,
    clashes: list,
    constraints: dict,
    program: str,
    semester: Optional[int],
    timetable_key: Optional[str],
    log_container,
    progress_bar,
    status_placeholder,
) -> tuple:
    log_queue: queue.Queue = queue.Queue()
    result_holder: dict = {}

    def _enqueue(line: str) -> None:
        log_queue.put(line)

    def _live_callback(turn: int, response: Any) -> None:
        for line in format_turn_log_entry(turn, response):
            _enqueue(line)
        progress_value = min(int((turn / 10) * 90), 90)
        progress_bar.progress(progress_value)
        status_placeholder.markdown(
            f"🤖 **Agent turn {turn}/10** — analyzing clashes and applying tools..."
        )

    def _worker() -> None:
        clash_detector = clash_detector_cls(firebase_manager)
        repaired, summary, remaining, agent_log = run_agentic_clash_repair(
            firebase_manager=firebase_manager,
            genetic_algorithm=genetic_algorithm,
            schedule=schedule,
            clashes=clashes,
            constraints=constraints,
            program=program,
            semester=semester,
            clash_detector=clash_detector,
            api_key=get_anthropic_api_key(),
            on_turn_callback=_live_callback,
        )
        for line in agent_log:
            _enqueue(line)
        result_holder["repaired"] = repaired
        result_holder["summary"] = summary
        result_holder["remaining"] = remaining
        log_queue.put(None)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    def _stream_logs():
        while True:
            try:
                item = log_queue.get(timeout=0.15)
            except queue.Empty:
                if not worker.is_alive():
                    break
                yield "⏳ Waiting for agent response...\n"
                continue
            if item is None:
                break
            yield item + "\n"

    with log_container:
        st.markdown("#### AGENT LOG (streaming):")
        st.write_stream(_stream_logs)

    worker.join(timeout=120)
    progress_bar.progress(100)
    status_placeholder.markdown("✅ **Repair complete**")

    return (
        result_holder.get("repaired", schedule),
        result_holder.get("summary", {}),
        result_holder.get("remaining", clashes),
    )


def render_agent_tab(firebase_manager, clash_detector_cls, genetic_algorithm=None):
    """Render the 🤖 AI Agent tab defined in the implementation blueprint."""
    st.markdown("### 🤖 AI Agent — Autonomous Clash Repair")

    program, semester, timetable_key, schedule = _resolve_schedule_and_key()
    constraints = _build_constraints(firebase_manager, timetable_key)
    clashes = _detect_clashes(schedule, constraints) if schedule else []
    counts = categorize_clashes(clashes)

    st.markdown("#### Current Schedule Status:")
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        if counts["faculty"]:
            st.warning(f"⚠️  {counts['faculty']} faculty clash(es) detected")
        else:
            st.success("✅  0 faculty clashes")
    with status_col2:
        if counts["room"]:
            st.warning(f"⚠️  {counts['room']} room clash(es) detected")
        else:
            st.success("✅  0 room clashes")
    with status_col3:
        if counts["cross_semester"]:
            st.warning(f"⚠️  {counts['cross_semester']} cross-semester clash(es)")
        else:
            st.success("✅  0 cross-semester clashes")

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    run_repair = btn_col1.button("🚀 Run Agentic Repair", use_container_width=True)
    view_details = btn_col2.button("📋 View Clash Details", use_container_width=True)
    refresh_dashboard = btn_col3.button(
        "📊 Refresh Repair Dashboard", use_container_width=True
    )

    if view_details:
        st.session_state["agent_show_clash_details"] = True

    if st.session_state.get("agent_show_clash_details"):
        st.markdown("#### 📋 Clash Details")
        if clashes:
            st.dataframe(pd.DataFrame(clashes), use_container_width=True, hide_index=True)
        else:
            st.info("No clashes detected in the current schedule.")

    log_container = st.container()
    progress_container = st.container()
    summary_container = st.container()

    if run_repair:
        if not schedule:
            st.error("No generated timetable found for the selected program/semester.")
        elif not clashes:
            st.success("✅ No clashes detected — schedule is already clean.")
        else:
            with progress_container:
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                status_placeholder.markdown("🔄 **Initializing agent repair...**")

            repaired, summary, remaining = _run_repair_with_streaming(
                firebase_manager=firebase_manager,
                clash_detector_cls=clash_detector_cls,
                genetic_algorithm=genetic_algorithm,
                schedule=schedule,
                clashes=clashes,
                constraints=constraints,
                program=program,
                semester=semester,
                timetable_key=timetable_key,
                log_container=log_container,
                progress_bar=progress_bar,
                status_placeholder=status_placeholder,
            )

            if timetable_key:
                st.session_state.setdefault("generated_schedules", {})[
                    timetable_key
                ] = repaired

            st.session_state["last_agent_session"] = summary
            st.session_state["last_agent_log"] = summary.get("conversation_log", [])

            with summary_container:
                render_repair_summary(summary)
                render_explain_repair_section(summary)
                if remaining:
                    st.warning(
                        f"⚠️ {len(remaining)} clash(es) remain after agent + fallback repair."
                    )
                else:
                    st.success("✅ Clash fixed! Schedule is now clean.")

    elif st.session_state.get("last_agent_session"):
        with summary_container:
            render_repair_summary(st.session_state["last_agent_session"])
            render_explain_repair_section(st.session_state["last_agent_session"])

    if refresh_dashboard or st.session_state.get("agent_show_dashboard", True):
        render_repair_history_dashboard(firebase_manager)
