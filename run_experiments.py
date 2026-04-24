import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from main import (
    DATASET_DIR,
    DEFAULT_MODEL,
    RESULTS_DIR,
    append_jsonl_record,
    build_run_record,
    dataset_file_paths,
    evaluate_case,
    load_cases_from_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch experiments for dataset cases."
    )
    parser.add_argument(
        "--cases-dir",
        help="Directory containing dataset JSON files. Defaults to the built-in dataset directory.",
    )
    parser.add_argument(
        "--case-file",
        action="append",
        default=[],
        help="Individual dataset JSON file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use. Defaults to '{DEFAULT_MODEL}'.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of runs per case. Defaults to 1.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help=f"Directory for run logs and batch summaries. Defaults to '{RESULTS_DIR.name}'.",
    )
    return parser.parse_args()


def discover_case_files(case_files: list[str], cases_dir: str | None) -> list[Path]:
    discovered: list[Path] = []

    for case_file in case_files:
        path = Path(case_file)
        if not path.exists() or not path.is_file():
            raise SystemExit(f"Case file not found: {case_file}")
        if path.suffix != ".json":
            raise SystemExit(f"Case file must be a JSON file: {case_file}")
        discovered.append(path)

    if cases_dir:
        directory = Path(cases_dir)
        if not directory.exists() or not directory.is_dir():
            raise SystemExit(f"Cases directory not found: {cases_dir}")
        discovered.extend(sorted(directory.glob("*.json")))
    elif not discovered:
        discovered.extend(dataset_file_paths(DATASET_DIR))

    unique_paths = sorted({path.resolve() for path in discovered})
    if not unique_paths:
        raise SystemExit("No dataset files specified or discovered.")
    return unique_paths


def summarize_records(records: list[dict[str, Any]], case_files: list[Path]) -> dict[str, Any]:
    attack_type_counts = Counter(record["attack_type"] for record in records)
    category_counts = Counter(record["category"] for record in records)
    action_counts = Counter(record["parsed_action"] for record in records)
    tool_counts = Counter(
        record["tool_name"] for record in records if record["tool_name"] is not None
    )

    total_runs = len(records)
    unauthorized_runs = sum(1 for record in records if record["unauthorized"])
    target_attack_hits = sum(1 for record in records if record["target_attack_hit"])
    expected_benign_runs = sum(1 for record in records if record["expected_benign_success"])
    benign_successful_runs = sum(1 for record in records if record["benign_success"])

    return {
        "total_runs": total_runs,
        "total_case_files": len(case_files),
        "total_case_entries": len({record["case_id"] for record in records}),
        "unauthorized_runs": unauthorized_runs,
        "target_attack_hits": target_attack_hits,
        "benign_expected_runs": expected_benign_runs,
        "benign_successful_runs": benign_successful_runs,
        "tasr_overall": unauthorized_runs / total_runs if total_runs else None,
        "target_attack_rate": target_attack_hits / total_runs if total_runs else None,
        "benign_success_rate": (
            benign_successful_runs / expected_benign_runs if expected_benign_runs else None
        ),
        "counts_by_attack_type": dict(attack_type_counts),
        "counts_by_category": dict(category_counts),
        "counts_by_action": dict(action_counts),
        "counts_by_tool_name": dict(tool_counts),
        "case_files": [str(path) for path in case_files],
    }


def write_summary_file(
    summary: dict[str, Any], results_dir: Path, batch_id: str
) -> Path:
    batch_dir = results_dir / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_path = batch_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=True, indent=2)
        file.write("\n")
    return summary_path


def print_summary(summary: dict[str, Any], summary_path: Path, runs_path: Path) -> None:
    print(f"Case files: {summary['total_case_files']}")
    print(f"Case entries: {summary['total_case_entries']}")
    print(f"Runs: {summary['total_runs']}")
    print(f"Unauthorized runs: {summary['unauthorized_runs']}")
    print(f"Target attack hits: {summary['target_attack_hits']}")
    print(f"Benign success rate: {summary['benign_success_rate']}")
    print(f"T-ASR overall: {summary['tasr_overall']}")
    print(f"Run log: {runs_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1.")

    case_files = discover_case_files(args.case_file, args.cases_dir)
    results_dir = Path(args.results_dir)

    records: list[dict[str, Any]] = []
    runs_path: Path | None = None
    for case_file in case_files:
        try:
            cases = load_cases_from_path(case_file)
        except ValueError as error:
            raise SystemExit(f"Invalid case file {case_file}: {error}") from error
        for case_data in cases:
            for _ in range(args.repeat):
                result = evaluate_case(case_data, args.model)
                record = build_run_record(result, case_data, args.model)
                runs_path = append_jsonl_record(record, results_dir)
                records.append(record)

    assert runs_path is not None
    batch_id = uuid4().hex
    summary = summarize_records(records, case_files)
    summary.update(
        {
            "batch_id": batch_id,
            "model": args.model,
            "repeat": args.repeat,
            "results_dir": str(results_dir),
            "runs_log": str(runs_path),
        }
    )
    summary_path = write_summary_file(summary, results_dir, batch_id)
    print_summary(summary, summary_path, runs_path)


if __name__ == "__main__":
    main()
