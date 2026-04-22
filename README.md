# Agentic Red Teaming Baseline

Minimaler Python-Runner fuer Issue-Triage-Experimente mit simulierten Tool-Calls.

## Voraussetzungen

- Python `>=3.12`
- Ein lokaler Ollama-Daemon
- Ein verfuegbares Ollama-Modell, standardmaessig `mistral`

## Setup

Mit `uv`:

```bash
uv sync
```

Alternativ mit `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Modell konfigurieren

Per CLI:

```bash
python3 main.py --model mistral /path/to/case.json
```

Oder per Umgebungsvariable:

```bash
export OLLAMA_MODEL=mistral
```

## Befehle

Mocks auflisten:

```bash
python3 main.py --list-mocks
```

Einzelfall ausfuehren:

```bash
python3 main.py /path/to/case.json
```

Alle verfuegbaren Mocks ausfuehren:

```bash
python3 main.py --eval
```

JSONL-Logs landen standardmaessig in `results/runs.jsonl`. Ein anderer Zielordner ist ueber `--results-dir` moeglich.

## Hinweise

- Wenn `data/` ausgelagert wurde, gibt `--list-mocks` kontrolliert `No mock cases found.` aus.
- Fuer `--eval` muss das Dataset wieder vorhanden sein.
- Jeder erfolgreiche Run schreibt genau eine JSONL-Zeile mit Modell, Prompt-Version, Aktion, Tool-Daten und Ergebnis.
