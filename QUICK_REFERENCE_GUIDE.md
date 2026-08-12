# QUICK REFERENCE GUIDE
## Smart Classroom & Timetable Scheduling System

---

## PROJECT AT A GLANCE

| Aspect | Details |
|--------|---------|
| **Name** | Agentic Autonomous Timetable Scheduling System |
| **Type** | Capstone project (college timetable generation + autonomous repair) |
| **Status** | Production-ready ✅ |
| **Codebase** | 41 Python files, ~18,000 LOC |
| **Tech Stack** | Streamlit, Firebase, Claude/Gemini LLM, Genetic Algorithm |
| **Testing** | 66/66 tests passing |
| **Deployment** | Docker, GitHub Actions, Streamlit Cloud ready |

---

## KEY INNOVATION

```
PROBLEM:        College timetable generation leaves 20-30% clashes
                Manual resolution takes 3-5 days

TRADITIONAL:    Pure GA / CSP solver → Final timetable (one-shot)
                Success: 60-70% clash-free

THIS SYSTEM:    Hybrid GA (fast) + LLM Agent (intelligent)
                ├─ Phase 1: Generate initial timetable (GA)
                ├─ Phase 2: Detect clashes (ClashAnalyzer)
                ├─ Phase 3: Repair autonomously (Claude ReAct agent)
                └─ Phase 4: Fallback + optimize (GA repair rounds)
                
                Success: 90%+ clash-free
                Time: ~4-5 minutes per timetable
```

---

## PROJECT STRUCTURE (ESSENTIAL FILES)

```
D:\sts\College/

MAIN APPLICATION:
  app.py                          (9,156 lines - Streamlit UI + orchestration)
  genetic_algorithm.py            (1,920 lines - GA + clash detection)
  constraint_engine.py            (296 lines - Hard/soft constraint evaluation)

AGENTIC AI LAYER (NEW):
  agent/
    ├─ timetable_agent.py         (ReAct loop orchestrator)
    ├─ tools.py                   (11 tool definitions)
    ├─ integration.py             (Streamlit ↔ Agent bridge)
    ├─ memory.py                  (Session state + Firebase persistence)
    ├─ prompts.py                 (System & user prompts for LLM)
    ├─ edge_cases.py              (Error handling, retry, revert guard)
    ├─ gemini_wrapper.py          (Gemini REST API wrapper)
    ├─ firebase_ops.py            (Agent ↔ Firebase interface)
    ├─ input_validation.py        (Tool argument validation)
    ├─ rate_limiter.py            (API usage rate limiting)
    ├─ agent_ui.py                (Streamlit UI for agent)
    ├─ metrics_collector.py       (Statistical analysis)
    ├─ research_export.py         (Paper artifact generation)
    └─ explain_repair.py          (LLM-based repair explanations)

CONFIGURATION:
  config/
    ├─ settings.py                (Tunable settings: max_turns, retry, etc.)
    └─ __init__.py

UTILITIES:
  utils/
    ├─ clash_analyzer.py          (Clash detection pipeline)
    ├─ logging_config.py          (Structured logging + Sentry)
    ├─ time_utils.py              (Time slot utilities)
    └─ interval_utils.py          (Interval overlap detection)

TESTING (11 test files):
  tests/
    ├─ test_agent_e2e.py          (End-to-end agent repair)
    ├─ test_agent_tools.py        (Individual tool testing)
    ├─ test_agent_firebase.py     (Firebase CRUD)
    ├─ test_phase2_integration.py (GA + agent integration)
    ├─ test_phase3_cases.py       (Complex scenarios)
    ├─ test_phase3_exports.py     (Research exports)
    ├─ test_phase4_edge_cases.py  (Error handling)
    ├─ test_phase4_demo_prep.py   (Demo readiness)
    ├─ test_improvements.py       (Code quality)
    ├─ agent_test_helpers.py      (Test utilities)
    └─ __pycache__

SCRIPTS:
  scripts/
    ├─ health_check.py            (System health verification)
    ├─ run_phase3_research.py     (Research metrics)
    └─ validate_demo_walkthrough.py

DEPLOYMENT:
  Dockerfile                       (Container image)
  .dockerignore
  .github/workflows/ci.yml         (GitHub Actions CI)
  requirements.txt                 (Dependencies)

DOCUMENTATION:
  README.md                        (Project overview)
  LICENSE                          (MIT)
  CONTRIBUTING.md                  (Contribution guidelines)
  IMPLEMENTATION_BLUEPRINT_AND_PRD.md (Design specification)
  PROJECT_COMPREHENSIVE_ANALYSIS.md  (This analysis)
  ARCHITECTURE_DEEP_DIVE.md        (Technical deep dive)

DATA:
  Datasets/                        (Pre-loaded test data)
  Research Paper/                  (6 analyzed papers)
  research_output/                 (Phase 3 metrics + figures)
```

---

## CORE ALGORITHMS CHEAT SHEET

### 1. Hybrid Genetic Algorithm (Phase 1)

```python
# Pipeline: Hungarian → Graph Coloring → GA Evolution

step_1_hungarian_assignment()
  ├─ Input: faculties, subjects
  └─ Output: faculty-subject pairs (optimal bipartite matching)

step_2_graph_coloring()
  ├─ Input: subjects, time slots
  ├─ Build conflict graph (edges = subject conflicts)
  └─ Output: slot assignments (greedy coloring)

step_3_genetic_algorithm()
  ├─ Population size: 20
  ├─ Generations: 50
  ├─ Selection: Tournament (k=3)
  ├─ Crossover: Single-point
  ├─ Mutation: 10% random slot swap
  └─ Fitness: (1 - clash_ratio) * 100 - soft_penalty * 0.5

Output: Best individual (timetable with ~70-85% clash-free rate)
```

### 2. Agentic AI Repair (Phase 3)

```python
# ReAct Loop: Reason → Act → Observe → Reflect

while (clashes > 0 AND turns < 10):
    
    REASON:
      ├─ LLM thinks: "Which clash to fix first?"
      ├─ "What's the best alternative slot?"
      └─ "What tool should I call?"
    
    ACT:
      ├─ LLM calls: tool_move_class() / tool_swap_classes()
      └─ Tool execution with REVERT GUARD:
         If (clashes_after > clashes_before):
            Restore pre-execution state
         Else:
            Keep change
    
    OBSERVE:
      └─ Tool result: success/failure, clashes_before/after
    
    REFLECT:
      ├─ Clash count decreased? → Continue
      ├─ No progress for 3 turns? → Escalate
      └─ Max turns reached? → Stop

Output: Repaired timetable + repair audit trail
```

### 3. Fallback Intelligent Repair (Phase 3 fallback)

```python
# Triggered if LLM fails or max turns exceeded

for round in range(25):
    for clash in clashes:
        ├─ Find free slot for conflicting class
        ├─ Move class to free slot
        ├─ Verify no new clash
        └─ Repeat for all clashes
```

---

## TOOL REGISTRY (11 TOOLS)

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| **tool_read_schedule** | Read timetable | school_key, batch_key | Full schedule |
| **tool_read_clashes** | Get clash list | (none) | Clashes array |
| **tool_move_class** | Move 1 class | school, batch, class, from_slot, to_slot | Success? |
| **tool_swap_classes** | Swap 2 classes | school, batch, class1, slot1, class2, slot2 | Success? |
| **tool_check_faculty_free** | Check faculty availability | faculty, day, slot | Free? |
| **tool_check_room_free** | Check room availability | room, day, slot, school | Free? |
| **tool_get_free_slots** | Find available slots | school, (day_preference) | Free slots[] |
| **tool_apply_fix** | Advanced modifications | fix_type, params | Success? |
| **tool_verify_schedule** | Validate timetable | (none) | Valid? Warnings? |
| **tool_log_repair** | Audit trail | repair_dict | Repair ID |
| **tool_escalate** | Mark unsolvable | reason | Status updated |

---

## CONSTRAINTS (HARD + SOFT)

### Hard Constraints (MUST respect)

```
1. Faculty morning limit: max 2 classes at 9:00 AM per week
2. Lunch sacred: Never use 13:00-13:50 for lectures
3. Lab duration: Exactly 2 consecutive hours
4. Faculty availability: No double-booking
5. Room availability: No double-booking
6. Cross-semester conflict: Faculty not busy elsewhere
```

### Soft Constraints (Affect fitness)

```
1. Minimize consecutive classes (-0.1 penalty)
2. Faculty morning preference (+0.05 bonus)
3. Subject slot preference (+0.03 bonus)
4. Room utilization balance (-0.02 penalty)
```

---

## FIREBASE SCHEMA QUICKREF

```
/timetables/{program}_Sem{semester}/
  └─ schedule: { school: { batch: { day: { slot: class_info } } } }

/agent_sessions/{session_id}/
  ├─ status: [in_progress, completed, partial, failed, llm_failed, max_turns_exceeded]
  ├─ repairs_applied: [{ turn, tool, input, result, reverted }, ...]
  ├─ conversation: [{ role, content }, ...]
  └─ clashes_found / clashes_fixed / turns_taken

/repair_history/{repair_id}/
  ├─ session_id / tool_name / tool_input / tool_result
  ├─ clashes_before / clashes_after / reverted
  └─ timestamp
```

---

## KEY METHODS & ENTRY POINTS

### Main Orchestrators

```python
# app.py
SmartTimetableScheduler.generate_hybrid_timetable()
SmartTimetableScheduler._evolve_with_progress()

# agent/timetable_agent.py
TimetableAgent.repair_schedule()           # Main agent entry
TimetableAgent._repair_loop()              # ReAct loop

# agent/integration.py
run_agentic_clash_repair()                 # High-level wrapper
```

### Core Algorithm Classes

```python
# genetic_algorithm.py
GeneticAlgorithm.create_individual()
GeneticAlgorithm.evolve()
GeneticAlgorithm.evaluate_fitness()

# utils/clash_analyzer.py
ClashAnalyzer.detect_all_clashes()
ClashAnalyzer.detect_faculty_clashes()
ClashAnalyzer.detect_room_clashes()
ClashAnalyzer.count_clashes()

# constraint_engine.py
ConstraintEngine.is_slot_allowed()
ConstraintEngine.evaluate_fitness_penalty()
```

### Tool Execution

```python
# agent/tools.py
ToolRegistry.execute(tool_name, tool_args)
ToolRegistry.get_all_tools()

# agent/edge_cases.py
execute_tool_with_revert_guard()           # Prevents bad fixes
call_llm_with_retry()                      # Handles LLM failures
```

---

## TESTING QUICK COMMANDS

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agent_e2e.py -v

# Run with coverage
pytest tests/ --cov=agent --cov=genetic_algorithm

# Run health check
python scripts/health_check.py

# Validate demo prep
python scripts/validate_demo_walkthrough.py
```

---

## ENVIRONMENT VARIABLES

```bash
# LLM Configuration
ANTHROPIC_API_KEY=sk-...         # Claude API key
GEMINI_API_KEY=...               # Gemini API key (fallback)

# Agent Tuning
AGENT_MAX_TURNS=10               # ReAct loop max iterations
AGENT_MAX_REPAIRS_PER_HOUR=30    # Rate limit
AGENT_LLM_MAX_RETRIES=3          # Retry on transient failure
AGENT_LLM_RETRY_DELAY=0.5        # Delay between retries (seconds)

# GA Tuning
GA_MAX_ATTEMPTS=2000             # Evolution attempts
GA_REPAIR_ROUNDS=25              # Fallback repair iterations

# Monitoring
SENTRY_DSN=...                   # Sentry error tracking (optional)
HEALTH_CHECK_TIMEOUT=10          # Health check timeout (seconds)
```

---

## DEPLOYMENT CHECKLIST

### Local Development

- [ ] Clone repository
- [ ] `pip install -r requirements.txt`
- [ ] Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`
- [ ] Add Firebase credentials + API keys to `secrets.toml`
- [ ] `streamlit run app.py`

### Docker

- [ ] Build: `docker build -t college-scheduler:latest .`
- [ ] Run: `docker run -p 8501:8501 -v /path/to/secrets.toml:/app/.streamlit/secrets.toml college-scheduler:latest`

### GitHub Actions

- [ ] Add secrets to GitHub repo (ANTHROPIC_API_KEY, Firebase creds)
- [ ] Push to trigger `.github/workflows/ci.yml`
- [ ] Verify all tests pass

### Cloud (Streamlit Cloud)

- [ ] Push code to GitHub
- [ ] Connect at https://share.streamlit.io
- [ ] Add secrets in Streamlit dashboard
- [ ] Auto-deploys on git push

---

## COMMON ISSUES & FIXES

| Issue | Cause | Fix |
|-------|-------|-----|
| `UnicodeEncodeError` on Windows | Console encoding issue | Set `PYTHONIOENCODING=utf-8` |
| Firebase connection timeout | Network issue or invalid creds | Verify `secrets.toml` has correct JSON |
| `NameError: name 'ToolRegistry' is not defined` | Import missing | Add `from agent.tools import ToolRegistry` |
| Agent status = "llm_failed" | LLM API error | Check API key + rate limits |
| Clashes not decreasing | Agent not finding good moves | Increase `AGENT_MAX_TURNS` |
| `ModuleNotFoundError: plotly` | Dependency missing | `pip install plotly` |

---

## PERFORMANCE BENCHMARKS

```
Operation                       Typical Time
────────────────────────────────────────────
Parse dataset (CSV)             100-500 ms
GA generation (50 gen)          2-3 min
Clash detection                 300-800 ms
Agent repair (10 turns)         30-60 sec
  (3-5 sec per LLM call)
Firebase CRUD (save)            100-300 ms
Export to PDF                   1-2 sec
────────────────────────────────────────────
TOTAL (GA + agent + export)     ~4-5 min
```

---

## RESEARCH ARTIFACTS

```
research_output/
├─ metrics/
│  ├─ agentic_vs_fallback.csv       # Comparison table
│  └─ clash_resolution_rates.json
├─ figures/
│  ├─ repair_effectiveness.png      # Matplotlib
│  └─ tool_usage_distribution.png
└─ local_agent_sessions/
   └─ {session_id}.json             # Full audit trail
```

**Run research export:**
```bash
python scripts/run_phase3_research.py
```

---

## CAPSTONE TALKING POINTS

1. **Novel Approach**
   - Hybrid GA (fast) + LLM Agent (intelligent) = unique combo
   - Not just CSP solver, but agentic reasoning

2. **Technical Depth**
   - 18,000+ LOC, complex constraint engine
   - ReAct loop with tool-calling (sophisticated)
   - Firebase integration (scalable)

3. **Production Quality**
   - Error handling (retry, revert guard, fallback)
   - Structured logging + Sentry
   - Docker + GitHub Actions ready

4. **Research Rigor**
   - Statistical comparison (agentic vs fallback)
   - Reproducible artifacts
   - Full audit trail

5. **Demo Readiness**
   - Live Streamlit UI
   - Pre-loaded datasets
   - Walkthrough scripts

---

## USEFUL LINKS

- **Documentation:** `README.md`
- **Architecture:** `ARCHITECTURE_DEEP_DIVE.md`
- **Full Analysis:** `PROJECT_COMPREHENSIVE_ANALYSIS.md`
- **Design Blueprint:** `IMPLEMENTATION_BLUEPRINT_AND_PRD.md`
- **GitHub:** (not provided, but ready for push)

---

## CONTACT & SUPPORT

For questions about:
- **Code structure:** See `ARCHITECTURE_DEEP_DIVE.md`
- **Implementation details:** See `PROJECT_COMPREHENSIVE_ANALYSIS.md`
- **Design decisions:** See `IMPLEMENTATION_BLUEPRINT_AND_PRD.md`
- **Testing:** Run `pytest tests/ -v`
- **Health check:** Run `python scripts/health_check.py`

---

**Last Updated:** August 11, 2026  
**Project Status:** ✅ Production-Ready for Capstone Submission
