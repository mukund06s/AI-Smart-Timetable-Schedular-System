"""
Phase 4 demo prep tests: documentation artifacts and explain-repair helper.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.explain_repair import explain_repair_plain_english


class Phase4DemoPrepTests(unittest.TestCase):
    def test_demo_walkthrough_doc_exists(self):
        path = ROOT / "docs" / "DEMO_WALKTHROUGH.md"
        self.assertTrue(path.exists(), "docs/DEMO_WALKTHROUGH.md is required")
        content = path.read_text(encoding="utf-8")
        self.assertIn("Screen Capture", content)
        self.assertIn("Run Agentic Repair", content)

    def test_capstone_report_sections_doc_exists(self):
        path = ROOT / "docs" / "CAPSTONE_REPORT_SECTIONS.md"
        self.assertTrue(path.exists(), "docs/CAPSTONE_REPORT_SECTIONS.md is required")
        content = path.read_text(encoding="utf-8")
        for section in ("Abstract", "Introduction", "Methodology", "Results", "Conclusion"):
            self.assertIn(section, content)

    def test_explain_repair_template_fallback(self):
        repair = {
            "action_type": "move",
            "from_slot": {"day": "Monday", "slot": "09:00-10:00"},
            "to_slot": {"day": "Tuesday", "slot": "10:00-11:00"},
            "reason": "resolve faculty clash",
        }
        text = explain_repair_plain_english(repair)
        self.assertIn("Monday", text)
        self.assertIn("Tuesday", text)

    def test_agent_ui_exports_streaming_helpers(self):
        import importlib.util

        ui_path = ROOT / "agent" / "agent_ui.py"
        spec = importlib.util.spec_from_file_location("agent_ui_module", ui_path)
        agent_ui = importlib.util.module_from_spec(spec)
        source = ui_path.read_text(encoding="utf-8")
        self.assertIn("render_repair_history_dashboard", source)
        self.assertIn("render_explain_repair_section", source)
        self.assertIn("_run_repair_with_streaming", source)
        self.assertIn("st.write_stream", source)


if __name__ == "__main__":
    unittest.main()
