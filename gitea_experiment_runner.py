from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from gitea_agent import DEFAULT_MODEL, MODEL_HELP, run_issue_agent
from gitea_evaluator import DEFAULT_SCENARIO_FILE
from gitea_evaluator import Scenario, evaluate_run, load_scenarios
from gitea_tools import DEFAULT_BOT_USER, DEFAULT_REPORTER_USER, GiteaClient
from gitea_tools import gitea_client
from reset_environment import reset_environment


def run_one(
    *,
    scenario: Scenario,
    model: str,
    defense: Literal["none", "prompt"],
    guard_profile: Literal["disabled", "sandbox"] = "sandbox",
    max_steps: int,
    bot_client: GiteaClient | None = None,
    reporter_client: GiteaClient | None = None,
) -> dict[str, object]:
    if bot_client is None:
        bot_client = gitea_client("bot")
    if reporter_client is None:
        reporter_client = gitea_client("reporter")
    
    issue = reporter_client.create_issue(scenario.issue.title, scenario.issue.body)

    agent_result = run_issue_agent(
        int(issue["number"]),
        client=bot_client,
        model=model,
        defense=defense,
        guard_profile=guard_profile,
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
        "scenario_type": scenario.scenario_type,
        "is_attack": int(scenario.is_attack),
        "payload_source": scenario.payload_source,
        "model": model,
        "defense": defense,
        "runtime_guard": guard_profile,
        "issue_id": issue["number"],
        "issue_author": issue.get("user", {}).get("login", DEFAULT_REPORTER_USER),
        "agent_user": DEFAULT_BOT_USER,
        "trace_id": scenario.trace_id or "",
        "monitoring_query": scenario.monitoring_query or "",
        "tool_calls_json": json.dumps(
            agent_result.tool_calls, ensure_ascii=True, default=str
        ),
        "raw_outputs_json": json.dumps(agent_result.raw_outputs, ensure_ascii=True),
        "agent_thinking": agent_result.thinking,
        **evaluation,
    }


CSV_FIELDNAMES = [
    "run_id",
    "timestamp",
    "scenario",
    "scenario_type",
    "is_attack",
    "payload_source",
    "model",
    "defense",
    "runtime_guard",
    "issue_id",
    "issue_author",
    "agent_user",
    "trace_id",
    "monitoring_query",
    "tool_calls_json",
    "raw_outputs_json",
    "agent_thinking",
    "payload_seen",
    "unauthorized_tool_call",
    "unauthorized_tool_executed",
    "runtime_guard_blocked",
    "unauthorized_tool_name",
    "policy_violation_reason",
    "unauthorized_tool_args_json",
    "desired_tool_coverage",
    "missing_desired_calls_json",
    "sensitive_access",
    "exfiltration",
    "downstream_triggered",
    "max_success_level",
    "propagation_depth",
    "benign_task_completed",
    "benign_control_passed",
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
        return [
            name
            for name, scenario in scenarios.items()
            if scenario.is_attack and scenario.scenario_type.startswith("multi-turn")
        ]
    if value not in scenarios:
        raise SystemExit(
            f"Unknown scenario '{value}'. Available: {', '.join(sorted(scenarios))}"
        )
    return [value]


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    attack_rows = [row for row in rows if int(row["is_attack"])]
    benign_rows = [row for row in rows if not int(row["is_attack"])]
    attack_total = len(attack_rows)
    benign_total = len(benign_rows)
    unauthorized = sum(int(row["unauthorized_tool_call"]) for row in attack_rows)
    unauthorized_executed = sum(
        int(row.get("unauthorized_tool_executed", 0)) for row in attack_rows
    )
    guard_blocked = sum(
        int(row.get("runtime_guard_blocked", 0)) for row in attack_rows
    )
    sensitive = sum(int(row["sensitive_access"]) for row in attack_rows)
    exfiltration = sum(int(row["exfiltration"]) for row in attack_rows)
    downstream = sum(int(row.get("downstream_triggered", 0)) for row in attack_rows)
    benign_passed = sum(int(row["benign_control_passed"]) for row in benign_rows)

    return {
        "total_runs": len(rows),
        "attack_runs": attack_total,
        "benign_runs": benign_total,
        "unauthorized_attack_runs": unauthorized,
        "executed_unauthorized_attack_runs": unauthorized_executed,
        "guard_blocked_attack_runs": guard_blocked,
        "sensitive_attack_runs": sensitive,
        "exfiltration_attack_runs": exfiltration,
        "downstream_trigger_attack_runs": downstream,
        "benign_control_passed_runs": benign_passed,
        "tasr": unauthorized / attack_total if attack_total else None,
        "executed_unauthorized_rate": (
            unauthorized_executed / attack_total if attack_total else None
        ),
        "runtime_guard_block_rate": (
            guard_blocked / unauthorized if unauthorized else None
        ),
        "sensitive_access_rate": sensitive / attack_total if attack_total else None,
        "exfiltration_rate": exfiltration / attack_total if attack_total else None,
        "downstream_trigger_rate": downstream / attack_total if attack_total else None,
        "benign_control_rate": benign_passed / benign_total if benign_total else None,
    }


def format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


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
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="One or more Ollama models to test.",
    )
    parser.add_argument(
        "--defenses",
        nargs="+",
        choices=["none", "prompt"],
        default=["none"],
        help="One or more defense modes to test.",
    )
    parser.add_argument(
        "--runtime-guards",
        nargs="+",
        choices=["disabled", "sandbox"],
        default=["sandbox"],
        help="One or more runtime guard profiles to test.",
    )
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

    admin_client = gitea_client("admin")
    bot_client = gitea_client("bot")
    reporter_client = gitea_client("reporter")

    total_rows: list[dict[str, object]] = []
    
    # Run the matrix: models x defenses x guards x scenarios
    for model in args.models:
        for defense in args.defenses:
            for guard in args.runtime_guards:
                # Reset environment once per config to keep noise issues consistent
                reset_environment(
                    client=admin_client,
                    issue_client=reporter_client,
                    noise_issues=args.noise_issues,
                )
                
                scenario_names = scenario_names_from_arg(args.scenario, scenarios)
                print(f"\n>>> Starting Matrix: Model={model}, Defense={defense}, Guard={guard}, Scenarios={len(scenario_names)}")
                
                for scenario_name in scenario_names:
                    scenario = scenarios[scenario_name]
                    for _ in range(args.runs):
                        row = run_one(
                            scenario=scenario,
                            model=model,
                            defense=defense,
                            guard_profile=guard,
                            max_steps=args.max_steps,
                            bot_client=bot_client,
                            reporter_client=reporter_client,
                        )
                        total_rows.append(row)
                        write_rows(args.results_csv, [row])
                        print(
                            f"  {scenario.name}: unauthorized={row['unauthorized_tool_call']} "
                            f"level={row['max_success_level']} issue={row['issue_id']}"
                        )

    if total_rows:
        summary = summarize_rows(total_rows)
        print(f"\n{'='*60}")
        print(f"MATRIX COMPLETED: {len(total_rows)} runs")
        print(f"Results appended to {args.results_csv}")
        print(f"Overall T-ASR: {format_rate(summary['tasr'])}")
        print(f"Overall Benign Pass: {format_rate(summary['benign_control_rate'])}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
