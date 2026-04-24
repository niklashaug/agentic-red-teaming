# Agentic Red Teaming Baseline

Minimal Python runner for issue-triage experiments with simulated tool calls and a typed dataset schema.

## Requirements

- Python `>=3.12`
- A local Ollama daemon
- An available Ollama model, default: `mistral`

## Setup

Using `uv`:

```bash
uv sync
```

Alternatively with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset Layout

The canonical dataset lives under `dataset/`:

- `dataset/single_turn_attack.json`
- `dataset/multi_turn_attack.json`
- `dataset/benign.json`

Each file contains a JSON array of cases with this schema:

```json
{
  "id": "case_01",
  "category": "data_exfiltration",
  "attack_type": "single_turn",
  "user_task": "Summarize the issue and assign a priority label.",
  "issue": {
    "title": "...",
    "description": "...",
    "comments": ["...", "..."]
  },
  "expected_behavior": {
    "allowed_actions": [
      {
        "tool": "add_label",
        "arguments": {"label": "bug"}
      }
    ],
    "forbidden_actions": [
      {
        "tool": "read_environment_variable",
        "arguments": {"name": "OPENAI_API_KEY"}
      }
    ]
  },
  "evaluation": {
    "target_attack": "read_environment_variable:OPENAI_API_KEY",
    "expected_benign_success": true
  }
}
```

## Commands

List all built-in case ids:

```bash
python3 main.py --list-cases
```

Run a single built-in case by id:

```bash
python3 main.py single_turn_exfiltration_01
```

Run a single external case file:

```bash
python3 main.py /path/to/case.json
```

Run all built-in dataset cases:

```bash
python3 main.py --eval
```

Batch run against the built-in dataset:

```bash
python3 run_experiments.py --model mistral --repeat 3
```

Batch run against another dataset directory:

```bash
python3 run_experiments.py --cases-dir /path/to/dataset --repeat 5
```

JSONL logs are written to `results/runs.jsonl` by default. Batch aggregations are written to `results/batches/<batch_id>/summary.json`.

## Tests

```bash
python3 -m unittest discover -s tests
```

## Notes

- `main.py` now expects the typed dataset schema and does not support the older flat `issue_id` / `issue_text` format.
- `expected_behavior.forbidden_actions` is used to detect unauthorized tool calls.
- `evaluation.target_attack` is tracked separately from generic unauthorized access so the dataset can express a specific attack objective.
