# Contributing

Thank you for contributing to the Smart Timetable Scheduler project.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and configure credentials
4. Run tests before submitting changes: `python -m unittest discover -s tests`

## Code Standards

- Preserve existing functionality; avoid unrelated refactors
- Add tests for new agent tools, integration paths, or utilities
- Use structured logging via `utils.logging_config.get_logger(__name__)`
- Catch specific exceptions; never use bare `except:`
- Keep agent tool validation in `agent/input_validation.py`

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] Health check passes: `python scripts/health_check.py`
- [ ] No secrets committed
- [ ] README updated if setup or configuration changed

## Reporting Issues

Include: steps to reproduce, expected vs actual behavior, program/semester context, and relevant log output.
