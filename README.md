# Smart Classroom & Timetable Scheduler

Agentic AI-powered college timetable generation and autonomous clash repair system.

## Overview

Hybrid scheduling pipeline:

1. **Hybrid Genetic Algorithm** — initial timetable generation (Hungarian assignment, graph coloring, GA evolution)
2. **Clash Detection** — intra-schedule, room, and cross-semester faculty conflicts
3. **Agentic AI Repair** — Claude-based ReAct agent with 11 tools for autonomous clash resolution
4. **Firebase Persistence** — timetables, agent sessions, repair audit trail

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Database | Firebase Firestore |
| LLM Agent | Anthropic Claude |
| Optimization | SciPy, NetworkX, custom GA |
| Analytics | Plotly, Matplotlib, Pandas |

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd College
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Configure secrets
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# Edit secrets.toml with Firebase credentials and Anthropic API key

# 3. Run
streamlit run app.py
```

## Configuration

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude API key (alternative to secrets.toml) |
| `AGENT_MAX_REPAIRS_PER_HOUR` | 30 | Rate limit for agent repair sessions |
| `AGENT_MAX_TURNS` | 10 | Max LLM turns per repair |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `SENTRY_DSN` | — | Optional Sentry error tracking |

## Project Structure

```
College/
├── app.py                  # Main Streamlit application
├── genetic_algorithm.py    # Hybrid GA engine
├── agent/                  # Agentic AI layer
│   ├── timetable_agent.py  # ReAct repair loop
│   ├── tools.py            # 11 agent tools
│   ├── integration.py      # Pipeline hook + fallback
│   └── ...
├── utils/                  # Shared utilities
├── config/                 # Centralized settings
├── tests/                  # Automated test suite
├── scripts/                # CLI utilities
└── research_output/        # Exported metrics and figures
```

## Testing

```bash
python -m unittest discover -s tests -v
python scripts/health_check.py
python scripts/validate_demo_walkthrough.py
```

## Research & Demo

- Demo walkthrough: `docs/DEMO_WALKTHROUGH.md`
- Capstone report outline: `docs/CAPSTONE_REPORT_SECTIONS.md`
- Run benchmarks: Reports tab → Research Metrics & Paper Exports

## Docker

```bash
docker build -t timetable-scheduler .
docker run -p 8501:8501 timetable-scheduler
```

## License

MIT — see [LICENSE](LICENSE).
