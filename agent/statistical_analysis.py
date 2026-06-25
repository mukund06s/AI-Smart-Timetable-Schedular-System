"""
Statistical analysis helpers for research paper comparisons.
"""

from typing import Any, Dict, List, Optional

from agent.metrics_collector import ComparisonResult


def _try_ttest(a: List[float], b: List[float]) -> Dict[str, Any]:
    try:
        from scipy import stats

        if len(a) < 2 or len(b) < 2:
            return {
                "test": "paired_t_test",
                "available": False,
                "reason": "Need at least 2 samples per group",
            }
        statistic, p_value = stats.ttest_rel(a, b)
        return {
            "test": "paired_t_test",
            "available": True,
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "significant_at_0_05": bool(p_value < 0.05),
        }
    except ImportError:
        return {
            "test": "paired_t_test",
            "available": False,
            "reason": "scipy not installed",
        }


def analyze_comparison_results(
    results: List[ComparisonResult],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Build summary statistics and significance tests for legacy vs agentic repair.
    Uses scenario-level paired values when multiple scenarios are provided.
    """
    legacy_times = [r.legacy.repair_time_seconds for r in results]
    agentic_times = [r.agentic.repair_time_seconds for r in results]
    legacy_rates = [r.legacy.clash_resolution_rate_pct for r in results]
    agentic_rates = [r.agentic.clash_resolution_rate_pct for r in results]

    def _summary(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "mean": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    time_test = _try_ttest(legacy_times, agentic_times)
    rate_test = _try_ttest(legacy_rates, agentic_rates)

    return {
        "alpha": alpha,
        "scenarios_analyzed": len(results),
        "repair_time_seconds": {
            "legacy": _summary(legacy_times),
            "agentic": _summary(agentic_times),
            "mean_improvement_seconds": round(
                _summary(legacy_times)["mean"] - _summary(agentic_times)["mean"], 3
            ),
            "significance_test": time_test,
        },
        "resolution_rate_pct": {
            "legacy": _summary(legacy_rates),
            "agentic": _summary(agentic_rates),
            "mean_improvement_pct": round(
                _summary(agentic_rates)["mean"] - _summary(legacy_rates)["mean"], 3
            ),
            "significance_test": rate_test,
        },
        "interpretation": _interpret_results(time_test, rate_test, alpha),
    }


def _interpret_results(
    time_test: Dict[str, Any],
    rate_test: Dict[str, Any],
    alpha: float,
) -> str:
    parts = []
    if time_test.get("available") and time_test.get("significant_at_0_05"):
        parts.append(
            f"Repair time difference is statistically significant (p={time_test['p_value']}, alpha={alpha})."
        )
    elif time_test.get("available"):
        parts.append(
            f"Repair time difference is not statistically significant (p={time_test['p_value']})."
        )

    if rate_test.get("available") and rate_test.get("significant_at_0_05"):
        parts.append(
            f"Resolution rate difference is statistically significant (p={rate_test['p_value']})."
        )
    elif rate_test.get("available"):
        parts.append(
            f"Resolution rate difference is not statistically significant (p={rate_test['p_value']})."
        )

    if not parts:
        return "Insufficient data or scipy unavailable for significance testing."
    return " ".join(parts)


def export_statistical_report(
    results: List[ComparisonResult],
    export_path: str,
) -> str:
    """Write statistical analysis JSON for research paper appendix."""
    import json
    from pathlib import Path

    report = analyze_comparison_results(results)
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return str(path)
