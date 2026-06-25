#!/usr/bin/env python3
"""
Validate Phase 4 demo prep artifacts exist and are non-empty.
Run: python scripts/validate_demo_walkthrough.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "docs" / "DEMO_WALKTHROUGH.md",
    ROOT / "docs" / "CAPSTONE_REPORT_SECTIONS.md",
    ROOT / "agent" / "edge_cases.py",
    ROOT / "agent" / "explain_repair.py",
]


def main() -> int:
    missing = []
    for path in REQUIRED:
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        print("Missing or empty demo artifacts:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("Phase 4 demo prep artifacts validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
