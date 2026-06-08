# Agentic Red Teaming Gitea Benchmark

Local benchmark for indirect prompt injection in agentic issue-triage
workflows. The active setup uses a disposable Gitea repository, a local mock
log/monitoring server, and an Ollama-backed triage agent.

For the full user guide, see [docs/GITEA_SETUP.md](docs/GITEA_SETUP.md).

## Requirements

- Python `>=3.12`
- Docker Compose
- A local Ollama daemon
- An available Ollama tool-calling model, default: `qwen3:8b`

## Setup

Install Python dependencies:

```bash
uv sync
```

Pull the default tool-calling model:

```bash
ollama pull qwen3:8b
```

Start Gitea and the mock log server:

```bash
docker compose up -d --build
```

Create local Gitea users and role-specific PATs:

```bash
uv run python scripts/bootstrap_gitea.py
```

Reset the disposable test repository:

```bash
uv run python reset_environment.py
```

Run the automated benchmark loop:

```bash
uv run python gitea_experiment_runner.py --scenario all --runs 1 --defense none
```

## Active Files

- `dataset/gitea_scenarios.json`: attack and benign-control scenarios
- `docker/logserver/telemetry.json`: simulated log and monitoring artifacts
- `gitea_agent.py`: issue-triage agent loop
- `gitea_evaluator.py`: deterministic attack-success evaluator
- `gitea_experiment_runner.py`: reset, issue creation, agent execution, CSV logging
- `gitea_tools.py`: Gitea and logserver API wrappers

CSV output is appended to `results/gitea_experiments.csv`.

## Tests

```bash
python3 -m unittest discover -s tests
```
