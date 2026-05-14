"""
Run all GA configurations for one selected instance.

Examples:
  python run_instance_experiments.py --input data/input/kosovo_tv_input.json
  python run_instance_experiments.py --input kosovo_tv_input.json --seeds 1 2 3
  python run_instance_experiments.py
"""

import argparse
import csv
from pathlib import Path

from parser.file_selector import select_file
from run_experiments import (
    CONFIGS,
    LOCAL_SEARCH_BUDGET,
    RESULTS_DIR,
    SEEDS,
    TIME_LIMIT,
    append_result_csvs,
    run_single,
)


def resolve_instance_path(raw_input: str | None) -> str:
    if raw_input is None:
        return select_file()

    candidate = Path(raw_input)
    if candidate.exists():
        return str(candidate)

    input_candidate = Path("data/input") / raw_input
    if input_candidate.exists():
        return str(input_candidate)

    raise FileNotFoundError(
        f"Instance not found: {raw_input}. Use a full path or a file from data/input/."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all GA parameter configurations for a single instance."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Instance path or filename from data/input. If omitted, opens selector.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Optional subset of configs. Default: all configs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help="Seeds to run. Default: 1..10.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT,
        help="Total budget per run in seconds. Default: 300, max 600.",
    )
    parser.add_argument(
        "--ls-budget",
        type=float,
        default=LOCAL_SEARCH_BUDGET,
        help="Seconds reserved for Local Search after GA. Default: 30.",
    )
    parser.add_argument(
        "--guided-ls",
        action="store_true",
        help="Enable experimental Guided Local Search during the LS phase.",
    )
    return parser.parse_args()


def write_summary(rows: list[dict]) -> None:
    by_config: dict[str, list[dict]] = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)

    summary_rows = []
    print("\nSUMMARY FOR SELECTED INSTANCE")
    print("=" * 54)
    print(f"{'Config':<12} {'Runs':>5} {'GA best':>8} {'GA+LS best':>12} {'Best gain':>9}")
    print("-" * 54)

    for cfg_name, cfg_rows in by_config.items():
        ga_scores = [int(r["ga_score"]) for r in cfg_rows]
        ga_ls_scores = [int(r["ga_ls_score"]) for r in cfg_rows]
        ga_best = max(ga_scores)
        ga_ls_best = max(ga_ls_scores)
        summary = {
            "instance": cfg_rows[0]["instance"],
            "config": cfg_name,
            "ga_best": ga_best,
            "ga_ls_best": ga_ls_best,
            "best_gain": ga_ls_best - ga_best,
            "runs": len(cfg_rows),
        }
        summary_rows.append(summary)
        print(
            f"{cfg_name:<12} {summary['runs']:>5} {summary['ga_best']:>8} "
            f"{summary['ga_ls_best']:>12} {summary['best_gain']:>9}"
        )

    best = max(summary_rows, key=lambda row: row["ga_ls_best"])
    print("-" * 54)
    print(f"Best config by GA+LS best: {best['config']} (best={best['ga_ls_best']})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    instance_name = summary_rows[0]["instance"] if summary_rows else "selected_instance"
    summary_path = RESULTS_DIR / f"{instance_name}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["instance", "config", "ga_best", "ga_ls_best", "best_gain", "runs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary saved to {summary_path}")


def main():
    args = parse_args()
    instance_path = resolve_instance_path(args.input)
    instance_name = Path(instance_path).stem.replace("_input", "")

    if args.configs:
        unknown = [name for name in args.configs if name not in CONFIGS]
        if unknown:
            raise SystemExit(
                f"Unknown config(s): {unknown}. Available: {list(CONFIGS.keys())}"
            )
        configs = {name: CONFIGS[name] for name in args.configs}
    else:
        configs = dict(CONFIGS)

    seeds = args.seeds if args.seeds else list(SEEDS)
    time_limit = min(max(1.0, float(args.time_limit)), 600.0)
    ls_budget = max(0.0, float(args.ls_budget))

    total_runs = len(configs) * len(seeds)
    run_idx = 0
    rows: list[dict] = []

    print(f"\nInstance: {instance_path}")
    print(f"Configs: {', '.join(configs.keys())}")
    print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"Time limit: {time_limit}s | LS budget: {ls_budget}s\n")

    for cfg_name, cfg in configs.items():
        for seed in seeds:
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] {cfg_name} | {instance_name} | seed={seed}",
                end=" ... ",
                flush=True,
            )
            try:
                res = run_single(
                    instance_path,
                    seed,
                    cfg,
                    cfg_name,
                    time_limit,
                    ls_budget,
                    guided_local_search=args.guided_ls,
                )
                print(
                    f"GA={res['ga_score']}  GA+LS={res['ga_ls_score']}  "
                    f"imp={res['improvement']}  time={res['time_s']}s"
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                res = {
                    "ga_score": 0,
                    "ga_segments": 0,
                    "ga_ls_score": 0,
                    "ga_ls_segments": 0,
                    "improvement": 0,
                    "improvement_pct": 0.0,
                    "score": 0,
                    "segments": 0,
                    "GA+LS": 0,
                    "ga_time_s": 0,
                    "ls_time_s": 0,
                    "time_s": 0,
                    "ls_attempts": 0,
                    "ls_improvements": 0,
                }

            row = {
                "config": cfg_name,
                "instance": instance_name,
                "seed": seed,
                **res,
            }
            rows.append(row)
            append_result_csvs(row)

    write_summary(rows)
    print("\nResults saved to data/output and results/*.csv")


if __name__ == "__main__":
    main()
