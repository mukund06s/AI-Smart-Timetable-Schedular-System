# Demo Walkthrough — College Timetable Agentic AI Capstone

Use this script for live demo, viva, and screen-capture submission.

---

## Prerequisites

1. Python environment with project dependencies installed (`pip install -r requirements.txt`).
2. Firebase credentials configured (or demo mode with local backups enabled).
3. Anthropic API key in `.streamlit/secrets.toml` under `[agent] ANTHROPIC_API_KEY`.
4. Sample dataset loaded (BTECH / Semester 2 recommended).

---

## Demo Flow (15–20 minutes)

### Step 1 — Launch Application

```bash
streamlit run app.py
```

**Screen capture:** Show terminal launch and browser opening at `localhost:8501`.

### Step 2 — Load Data & Generate Timetable

1. Select **Program**: BTECH  
2. Select **Semester**: 2  
3. Upload or load the sample CSV dataset  
4. Navigate to **Generate Timetable** tab  
5. Click **Generate Hybrid Timetable**  
6. Wait for Phases 1–4 (GA + agent repair) to complete  

**Screen capture:** Progress bars for hybrid GA phases; note any clashes detected in Phase 4.

### Step 3 — Inspect Clashes

1. Open **Reports** tab → **Clash History Report**  
2. Show faculty/room/cross-semester clash counts  

**Talking point:** Hybrid GA may leave residual clashes; agent layer resolves them autonomously.

### Step 4 — Run Agentic Repair (AI Agent Tab)

1. Open **🤖 AI Agent** tab  
2. Review **Current Schedule Status** cards (faculty / room / cross-semester)  
3. Click **🚀 Run Agentic Repair**  
4. Watch **streaming agent log** (`st.write_stream`) and animated progress  
5. Review **REPAIR SUMMARY** (turns used, time, Firebase/local backup)  
6. Click **Explain repair #1** for plain-English LLM explanation  

**Screen capture:** Full agent log stream, progress indicator, summary panel, explain button output.

### Step 5 — Repair History Dashboard

1. Click **📊 Refresh Repair Dashboard**  
2. Show session metrics (total sessions, completed, clashes fixed)  
3. Filter sessions by status  
4. Scroll repair actions table  

**Screen capture:** Dashboard metrics and filtered session table.

### Step 6 — Research Metrics (Optional)

1. Reports tab → **📊 Research Metrics & Paper Exports**  
2. Click **Run Benchmarks & Export Bundle**  
3. Show comparison tables and exported CSV/figures in `research_output/`  

**Talking point:** Phase 3 research data supports capstone paper claims.

### Step 7 — Edge Cases (If Time Permits)

Demonstrate resilience:

| Scenario | How to Demo |
|----------|-------------|
| Max turns exceeded | Set `max_turns: 2` in Firebase `/agent_config/default`, run repair → fallback activates |
| LLM API failure | Temporarily invalidate API key → retry + fallback message in log |
| Firebase unavailable | Disconnect Firebase → local backup path shown in summary |
| Revert on bad fix | Agent log shows `↩️ repair(s) reverted` when fix would add clashes |

---

## Screen Capture Guidance for Submission

### Recording Settings

- **Resolution:** 1920×1080 (1080p)  
- **Duration:** 8–12 minutes (edited highlight reel) or full 15–20 min walkthrough  
- **Format:** MP4 (H.264)  
- **Audio:** Narrate each step; explain agent THOUGHT → ACTION → OBSERVATION loop  

### Required Shots (Checklist)

- [ ] App launch and program/semester selection  
- [ ] Timetable generation with hybrid GA progress  
- [ ] Clash detection report  
- [ ] AI Agent tab — Run Agentic Repair with streaming log  
- [ ] Animated progress bar during agent turns  
- [ ] Repair summary with metrics  
- [ ] "Explain this repair" button output  
- [ ] Repair history dashboard  
- [ ] (Optional) Research metrics export  

### Editing Tips

1. Add title slide: project name, team, date  
2. Use zoom/cursor highlight on agent log and summary panels  
3. Include 30-second architecture overview (Streamlit → GA → Agent → Firebase)  
4. End with results slide: clash resolution rate, turns used, fallback usage  

---

## Troubleshooting During Demo

| Issue | Fix |
|-------|-----|
| No timetable generated | Ensure dataset uploaded and program/semester selected |
| Agent tab empty | Generate timetable first; clashes required for demo |
| API key error | Check `.streamlit/secrets.toml` |
| Firebase errors | Local backup still saves session; mention graceful degradation |
| Slow LLM response | Pre-run once before live demo; use cached last session |

---

## Post-Demo Artifacts

After recording, attach:

1. Screen capture video (MP4)  
2. `research_output/` benchmark bundle  
3. Capstone report (see `CAPSTONE_REPORT_SECTIONS.md`)  
4. GitHub repository link  
