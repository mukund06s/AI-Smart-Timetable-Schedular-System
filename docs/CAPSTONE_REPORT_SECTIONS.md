# Capstone Report Sections — Agentic Timetable Scheduling System

Use this outline when writing the final capstone/project report. Fill each section with data from `research_output/` and demo recordings.

---

## 1. Abstract (150–250 words)

Summarize:

- Problem: automated college timetable generation with constraint satisfaction  
- Approach: Hybrid Genetic Algorithm + Agentic AI (Claude) autonomous clash repair  
- Key results: clash resolution rate, repair time, comparison vs legacy `_intelligent_repair`  
- Contribution: ReAct-style agent with 11 tools, Firebase audit trail, explainable repairs  

---

## 2. Introduction

### 2.1 Background

- Manual timetable scheduling challenges in colleges  
- NP-hard constraint satisfaction (faculty, rooms, batches, cross-semester)  

### 2.2 Problem Statement

- Residual clashes after GA optimization  
- Need for intelligent, explainable post-generation repair  

### 2.3 Objectives

1. Build hybrid GA scheduler (Phases 1–3 generation)  
2. Integrate agentic AI for autonomous clash resolution  
3. Persist repair history for audit and research  
4. Compare agentic vs legacy repair quantitatively  

### 2.4 Scope & Limitations

- Streamlit web UI; Firebase backend  
- Anthropic Claude API dependency  
- Single-institution demo dataset  

---

## 3. Literature Review

- Timetable scheduling algorithms (GA, graph coloring, Hungarian method)  
- Agentic AI and ReAct tool-calling patterns  
- Explainable AI in administrative systems  

---

## 4. System Architecture

### 4.1 High-Level Design

```
Streamlit UI → Hybrid GA → Clash Detection → TimetableAgent (ReAct) → Firebase
                     ↓                              ↓
              genetic_algorithm.py          agent/tools.py (11 tools)
```

### 4.2 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Database | Firebase Firestore |
| LLM | Anthropic Claude (claude-sonnet-4-5) |
| Optimization | genetic_algorithm.py, NetworkX, SciPy |
| Analytics | Plotly, Pandas |

### 4.3 Agent Tool Suite

List all 11 tools with purpose (read_clashes, move_class, swap_classes, etc.).

---

## 5. Methodology

### 5.1 Hybrid GA Pipeline

- Phase 1: Initial population  
- Phase 2: Crossover/mutation  
- Phase 3: Graph coloring + Hungarian assignment  
- Phase 4: Agentic clash repair (+ legacy fallback)  

### 5.2 Agent ReAct Loop

- THOUGHT → ACTION (tool call) → OBSERVATION → repeat  
- Max turns, retry on LLM failure, revert on clash increase  

### 5.3 Evaluation Scenarios

Reference Phase 3 scenarios A/B/C and six test cases from `agent/scenarios.py`.

### 5.4 Metrics

- Clash resolution rate  
- Repair time (seconds)  
- LLM turns used  
- Escalation count  
- Tool call frequency  
- Fallback usage rate  

---

## 6. Implementation

### 6.1 Phase 1 — Core Agent Infrastructure

- `agent/timetable_agent.py`, `agent/tools.py`, `agent/memory.py`  

### 6.2 Phase 2 — Pipeline Integration

- `agent/integration.py`, AI Agent tab, repair history in Reports  

### 6.3 Phase 3 — Testing & Research Data

- `agent/metrics_collector.py`, `agent/research_export.py`  
- CSV exports and paper figures in `research_output/`  

### 6.4 Phase 4 — Polish & Deployment

- Edge cases: max turns, LLM retry, Firebase local backup, revert guard  
- UI: streaming log, progress indicators, repair dashboard, explain repair  
- Demo walkthrough and screen capture (`docs/DEMO_WALKTHROUGH.md`)  

---

## 7. Results & Discussion

### 7.1 Quantitative Results

Insert Table 1 (Clash Resolution Comparison) and Table 2 (Time Complexity) from research exports.

### 7.2 Qualitative Observations

- Agent explainability via "Explain this repair"  
- Graceful degradation when API/Firebase unavailable  

### 7.3 Comparison: Agentic vs Legacy Repair

Discuss trade-offs: speed, resolution rate, auditability, API cost.

---

## 8. Testing

- Unit tests: `tests/test_agent_tools.py`, `tests/test_phase4_edge_cases.py`  
- Integration: `tests/test_phase2_integration.py`  
- E2E with mocked Anthropic: `tests/test_agent_e2e.py`  
- Report total test count and pass rate  

---

## 9. Conclusion

- Recap achievements  
- Agentic AI successfully resolves post-GA clashes with explainable audit trail  
- Future work: multi-agent coordination, real-time scheduling, cost optimization  

---

## 10. References

- Anthropic Claude API documentation  
- Genetic algorithm timetabling papers  
- ReAct: Synergizing Reasoning and Acting in Language Models  

---

## 11. Appendices

- **Appendix A:** Agent system prompt (`agent/prompts.py`)  
- **Appendix B:** Sample agent conversation log  
- **Appendix C:** Demo screen capture link  
- **Appendix D:** Firebase schema (`/agent_sessions`, `/repair_history`, `/agent_config`)  
