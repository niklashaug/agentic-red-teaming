from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from gitea_agent import DEFAULT_MODEL, run_issue_agent
from gitea_evaluator import DEFAULT_SCENARIO_FILE
from gitea_evaluator import Scenario, evaluate_run, load_scenarios
from gitea_tools import DEFAULT_BOT_USER, DEFAULT_REPORTER_USER
from gitea_tools import gitea_client
from reset_environment import reset_environment


def run_one(
    *,
    scenario: Scenario,
    model: str,
    defense: Literal["none", "prompt"],
    noise_issues: int,
    max_steps: int,
) -> dict[str, object]:
    admin_client = gitea_client("admin")
    bot_client = gitea_client("bot")
    reporter_client = gitea_client("reporter")
    reset_environment(
        client=admin_client,
        issue_client=reporter_client,
        noise_issues=noise_issues,
    )
    issue = reporter_client.create_issue(scenario.title, scenario.body)

    agent_result = run_issue_agent(
        int(issue["number"]),
        client=bot_client,
        model=model,
        defense=defense,
        max_steps=max_steps,
    )
    final_issue = bot_client.get_issue(int(issue["number"]))
    final_comments = bot_client.list_issue_comments(int(issue["number"]))
    evaluation = evaluate_run(
        scenario,
        agent_result.tool_calls,
        agent_result.final_action,
        issue_id=int(issue["number"]),
        final_issue=final_issue,
        final_comments=final_comments,
    )
    return {
        "run_id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario.name,
        "model": model,
        "defense": defense,
        "issue_id": issue["number"],
        "issue_author": issue.get("user", {}).get("login", DEFAULT_REPORTER_USER),
        "agent_user": DEFAULT_BOT_USER,
        "trace_id": scenario.trace_id or "",
        "monitoring_query": scenario.monitoring_query or "",
        "tool_calls_json": json.dumps(
            agent_result.tool_calls, ensure_ascii=True, default=str
        ),
        "raw_outputs_json": json.dumps(agent_result.raw_outputs, ensure_ascii=True),
        **evaluation,
    }


CSV_FIELDNAMES = [
    "run_id",
    "timestamp",
    "scenario",
    "model",
    "defense",
    "issue_id",
    "issue_author",
    "agent_user",
    "trace_id",
    "monitoring_query",
    "tool_calls_json",
    "raw_outputs_json",
    "payload_seen",
    "unauthorized_tool_call",
    "unauthorized_tool_name",
    "sensitive_access",
    "exfiltration",
    "max_success_level",
    "propagation_depth",
    "benign_task_completed",
    "final_label",
    "final_comment",
]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def scenario_names_from_arg(value: str, scenarios: dict[str, Scenario]) -> list[str]:
    if value in {"both", "all"}:
        return list(scenarios)
    if value == "multi-turn":
        return [name for name in scenarios if name.startswith("multi-turn")]
    if value not in scenarios:
        raise SystemExit(
            f"Unknown scenario '{value}'. Available: {', '.join(sorted(scenarios))}"
        )
    return [value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gitea-backed staged indirect prompt-injection experiments."
        )
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--scenario",
        default="all",
        help=(
            "Scenario name, 'multi-turn', or 'all'. Defaults to all scenarios "
            f"from {DEFAULT_SCENARIO_FILE}."
        ),
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=DEFAULT_SCENARIO_FILE,
        help=f"JSON scenario file. Defaults to {DEFAULT_SCENARIO_FILE}.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--defense", choices=["none", "prompt"], default="none")
    parser.add_argument("--noise-issues", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/gitea_experiments.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")

    scenarios = load_scenarios(args.scenario_file)
    rows: list[dict[str, object]] = []
    for scenario_name in scenario_names_from_arg(args.scenario, scenarios):
        scenario = scenarios[scenario_name]
        for _ in range(args.runs):
            row = run_one(
                scenario=scenario,
                model=args.model,
                defense=args.defense,
                noise_issues=args.noise_issues,
                max_steps=args.max_steps,
            )
            rows.append(row)
            write_rows(args.results_csv, [row])
            print(
                f"{scenario.name}: unauthorized={row['unauthorized_tool_call']} "
                f"level={row['max_success_level']} issue={row['issue_id']}"
            )

    total = len(rows)
    unauthorized = sum(int(row["unauthorized_tool_call"]) for row in rows)
    sensitive = sum(int(row["sensitive_access"]) for row in rows)
    exfiltration = sum(int(row["exfiltration"]) for row in rows)
    print(
        f"Wrote {total} rows to {args.results_csv}; "
        f"T-ASR={unauthorized / total:.3f}; "
        f"sensitive_access_rate={sensitive / total:.3f}; "
        f"exfiltration_rate={exfiltration / total:.3f}"
    )


if __name__ == "__main__":
    main()
