"""
Phase 3 research paper export utilities — logs, schedules, metrics CSV, figures.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agent.metrics_collector import ComparisonResult, MetricsCollector
from agent.statistical_analysis import export_statistical_report

DEFAULT_EXPORT_DIR = Path("research_output")


def ensure_export_dir(export_dir: Optional[Path] = None) -> Path:
    target = Path(export_dir or DEFAULT_EXPORT_DIR)
    target.mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(parents=True, exist_ok=True)
    (target / "schedules").mkdir(parents=True, exist_ok=True)
    (target / "logs").mkdir(parents=True, exist_ok=True)
    (target / "metrics").mkdir(parents=True, exist_ok=True)
    return target


def export_conversation_logs(
    conversation_log: List[dict],
    session_id: str,
    export_dir: Optional[Path] = None,
) -> str:
    base = ensure_export_dir(export_dir)
    path = base / "logs" / f"conversation_{session_id}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(conversation_log, handle, indent=2, default=str)
    return str(path)


def export_before_after_schedules(
    before_schedule: dict,
    after_schedule: dict,
    scenario_name: str,
    export_dir: Optional[Path] = None,
) -> Dict[str, str]:
    base = ensure_export_dir(export_dir)
    safe_name = scenario_name.replace(" ", "_").replace("/", "_")
    before_path = base / "schedules" / f"{safe_name}_before.json"
    after_path = base / "schedules" / f"{safe_name}_after.json"
    with open(before_path, "w", encoding="utf-8") as handle:
        json.dump(before_schedule, handle, indent=2, default=str)
    with open(after_path, "w", encoding="utf-8") as handle:
        json.dump(after_schedule, handle, indent=2, default=str)
    return {"before": str(before_path), "after": str(after_path)}


def export_metrics_csv(
    comparison_results: List[ComparisonResult],
    export_dir: Optional[Path] = None,
) -> Dict[str, str]:
    base = ensure_export_dir(export_dir)
    comparison_rows = MetricsCollector.build_comparison_table(comparison_results)
    time_rows = MetricsCollector.build_time_complexity_table(comparison_results)

    table1_path = base / "metrics" / "table1_clash_resolution_comparison.csv"
    table2_path = base / "metrics" / "table2_time_complexity.csv"
    combined_path = base / "metrics" / "research_metrics_combined.csv"

    pd.DataFrame(comparison_rows).to_csv(table1_path, index=False)
    pd.DataFrame(time_rows).to_csv(table2_path, index=False)

    combined = pd.DataFrame(comparison_rows + time_rows)
    combined.to_csv(combined_path, index=False)

    return {
        "table1_clash_resolution_comparison": str(table1_path),
        "table2_time_complexity": str(table2_path),
        "research_metrics_combined": str(combined_path),
    }


def generate_research_figures(
    comparison_results: List[ComparisonResult],
    export_dir: Optional[Path] = None,
) -> Dict[str, str]:
    base = ensure_export_dir(export_dir)
    figure_paths: Dict[str, str] = {}

    scenarios = [r.scenario_name for r in comparison_results]
    legacy_before = [r.legacy.clashes_at_start for r in comparison_results]
    legacy_after = [r.legacy.clashes_after_fix for r in comparison_results]
    agent_before = [r.agentic.clashes_at_start for r in comparison_results]
    agent_after = [r.agentic.clashes_after_fix for r in comparison_results]
    agent_turns = [r.agentic.iterations_or_turns for r in comparison_results]

    tool_counts: Dict[str, int] = {}
    for result in comparison_results:
        for tool, count in result.agentic.tool_call_counts.items():
            tool_counts[tool] = tool_counts.get(tool, 0) + count
    if not tool_counts:
        tool_counts = {"move": 1, "verify": 1, "log_repair": 1}

    figure_paths["figure1_clash_comparison"] = _save_figure_1(
        base, scenarios, legacy_before, legacy_after, agent_before, agent_after
    )
    figure_paths["figure2_turns_distribution"] = _save_figure_2(
        base, scenarios, agent_turns
    )
    figure_paths["figure3_tool_frequency"] = _save_figure_3(base, tool_counts)
    return figure_paths


def _save_figure_1(base, scenarios, legacy_before, legacy_after, agent_before, agent_after):
    path_html = base / "figures" / "figure1_before_after_clash_count.html"
    path_png = base / "figures" / "figure1_before_after_clash_count.png"

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_bar(name="Legacy Before", x=scenarios, y=legacy_before, marker_color="#EF553B")
        fig.add_bar(name="Legacy After", x=scenarios, y=legacy_after, marker_color="#FFA15A")
        fig.add_bar(name="Agentic Before", x=scenarios, y=agent_before, marker_color="#636EFA")
        fig.add_bar(name="Agentic After", x=scenarios, y=agent_after, marker_color="#00CC96")
        fig.update_layout(
            title="Figure 1: Before/After Clash Count per Scenario",
            barmode="group",
            xaxis_title="Scenario",
            yaxis_title="Clash Count",
        )
        fig.write_html(str(path_html))
        _try_write_png(fig, path_png)
    except Exception:
        _save_matplotlib_grouped_bar(
            path_png,
            scenarios,
            [
                ("Legacy Before", legacy_before),
                ("Legacy After", legacy_after),
                ("Agentic Before", agent_before),
                ("Agentic After", agent_after),
            ],
            "Figure 1: Before/After Clash Count per Scenario",
        )
    return str(path_html if path_html.exists() else path_png)


def _save_figure_2(base, scenarios, agent_turns):
    path_html = base / "figures" / "figure2_agent_turns_distribution.html"
    path_png = base / "figures" / "figure2_agent_turns_distribution.png"
    try:
        import plotly.express as px

        fig = px.bar(
            x=scenarios,
            y=agent_turns,
            labels={"x": "Scenario", "y": "Agent Turns Used"},
            title="Figure 2: Agent Turns Distribution",
            color=agent_turns,
            color_continuous_scale="Blues",
        )
        fig.write_html(str(path_html))
        _try_write_png(fig, path_png)
    except Exception:
        _save_matplotlib_bar(path_png, scenarios, agent_turns, "Figure 2: Agent Turns Distribution")
    return str(path_html if path_html.exists() else path_png)


def _save_figure_3(base, tool_counts):
    path_html = base / "figures" / "figure3_tool_call_frequency.html"
    path_png = base / "figures" / "figure3_tool_call_frequency.png"
    labels = list(tool_counts.keys())
    values = list(tool_counts.values())
    try:
        import plotly.express as px

        fig = px.pie(
            names=labels,
            values=values,
            title="Figure 3: Tool Call Frequency",
        )
        fig.write_html(str(path_html))
        _try_write_png(fig, path_png)
    except Exception:
        _save_matplotlib_pie(path_png, labels, values, "Figure 3: Tool Call Frequency")
    return str(path_html if path_html.exists() else path_png)


def _try_write_png(fig, path_png: Path) -> None:
    try:
        fig.write_image(str(path_png))
    except Exception:
        pass


def _save_matplotlib_grouped_bar(path_png, labels, series, title):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (name, values) in enumerate(series):
        ax.bar(x + (idx - 1.5) * width, values, width, label=name)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Clash Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def _save_matplotlib_bar(path_png, labels, values, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="#636EFA")
    ax.set_title(title)
    ax.set_ylabel("Turns")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def _save_matplotlib_pie(path_png, labels, values, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=labels, autopct="%1.1f%%")
    ax.set_title(title)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def export_research_bundle(
    comparison_results: List[ComparisonResult],
    conversation_logs: Optional[Dict[str, List[dict]]] = None,
    before_after_schedules: Optional[Dict[str, Dict[str, dict]]] = None,
    export_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export all Phase 3 research artifacts in one bundle."""
    base = ensure_export_dir(export_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    bundle_manifest: Dict[str, Any] = {
        "generated_at": timestamp,
        "export_dir": str(base),
        "metrics_csv": export_metrics_csv(comparison_results, base),
        "figures": generate_research_figures(comparison_results, base),
        "conversation_logs": {},
        "schedules": {},
        "screenshot_figures": [],
    }

    stats_path = base / "metrics" / f"statistical_analysis_{timestamp}.json"
    bundle_manifest["statistical_analysis"] = export_statistical_report(
        comparison_results, str(stats_path)
    )

    conversation_logs = conversation_logs or {}
    for session_id, log in conversation_logs.items():
        bundle_manifest["conversation_logs"][session_id] = export_conversation_logs(
            log, session_id, base
        )

    before_after_schedules = before_after_schedules or {}
    for scenario_name, schedules in before_after_schedules.items():
        bundle_manifest["schedules"][scenario_name] = export_before_after_schedules(
            schedules.get("before", {}),
            schedules.get("after", {}),
            scenario_name,
            base,
        )

    for _, figure_path in bundle_manifest["figures"].items():
        png_candidate = Path(figure_path).with_suffix(".png")
        if png_candidate.exists():
            bundle_manifest["screenshot_figures"].append(str(png_candidate))
        elif str(figure_path).endswith(".png") and Path(figure_path).exists():
            bundle_manifest["screenshot_figures"].append(str(figure_path))

    manifest_path = base / f"research_bundle_manifest_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(bundle_manifest, handle, indent=2, default=str)
    bundle_manifest["manifest"] = str(manifest_path)
    return bundle_manifest
