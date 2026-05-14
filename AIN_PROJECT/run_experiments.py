"""
Eksperimente: 4 konfigurime x 17 instanca x 10 seeds.
  python run_experiments.py
  python run_experiments.py --configs Intensiv Balancuar

Per cdo ekzekutim:
  - Printon GA dhe GA+LS ne console
  - Ruan JSON output vetem per GA+LS
  - Ruan CSV globale, CSV per konfigurim dhe CSV per instance
  - Ne fund printon dhe ruan summary
"""
import argparse
import csv
import json
import time
from pathlib import Path

from parser.parser import Parser
from scheduler.apex_upgraded_scheduler import ApexUpgradedScheduler
from utils.utils import Utils

INSTANCES = [
    "data/input/toy.json",
    "data/input/croatia_tv_input.json",
    "data/input/germany_tv_input.json",
    "data/input/kosovo_tv_input.json",
    "data/input/netherlands_tv_input.json",
    "data/input/uk_tv_input.json",
    "data/input/usa_tv_input.json",
    "data/input/australia_iptv.json",
    "data/input/france_iptv.json",
    "data/input/spain_iptv.json",
    "data/input/uk_iptv.json",
    "data/input/us_iptv.json",
    "data/input/singapore_pw.json",
    "data/input/canada_pw.json",
    "data/input/china_pw.json",
    "data/input/youtube_gold.json",
    "data/input/youtube_premium.json",
]

CONFIGS = {
    "Default": dict(
        population_size=60,
        crossover_rate=0.55,
        mutation_rate=0.50,
        tournament_k=5,
        elite_fraction=0.20,
        destroy_rebuild_rate=0.78,
        block_crossover_rate=0.88,
    ),
    "Explorues": dict(
        population_size=80,
        crossover_rate=0.40,
        mutation_rate=0.70,
        tournament_k=3,
        elite_fraction=0.10,
        destroy_rebuild_rate=0.78,
        block_crossover_rate=0.88,
    ),
    "Intensiv": dict(
        population_size=40,
        crossover_rate=0.80,
        mutation_rate=0.25,
        tournament_k=7,
        elite_fraction=0.30,
        destroy_rebuild_rate=0.78,
        block_crossover_rate=0.88,
    ),
    "Balancuar": dict(
        population_size=60,
        crossover_rate=0.65,
        mutation_rate=0.45,
        tournament_k=5,
        elite_fraction=0.20,
        destroy_rebuild_rate=0.90,
        block_crossover_rate=0.88,
    ),
}

SEEDS = list(range(1, 11))
TIME_LIMIT = 300.0
LOCAL_SEARCH_BUDGET = 30.0

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("data/output")

CSV_FIELDS = [
    "config",
    "instance",
    "seed",
    "score",
    "segments",
    "time_s",
    "GA+LS",
    "improvement",
]


def save_solution_json(solution, instance_path: str, cfg_name: str, seed: int, phase: str):
    """Save a GA or GA+LS solution in the same output format as the project."""
    if solution is None:
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = Path(instance_path).stem.replace("_input", "")
    safe_cfg = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in cfg_name)
    safe_phase = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in phase)
    score = int(solution.total_score)
    output_path = OUTPUT_DIR / f"{base_name}_output_{safe_cfg}_{safe_phase}_seed{seed}_{score}.json"

    schedules = [
        {
            "program_id": schedule.program_id,
            "channel_id": schedule.channel_id,
            "start": schedule.start,
            "end": schedule.end,
        }
        for schedule in solution.scheduled_programs
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"scheduled_programs": schedules}, f, indent=4, ensure_ascii=False)

    return output_path


def run_single(
    instance_path: str,
    seed: int,
    config: dict,
    cfg_name: str,
    time_limit: float,
    local_search_budget: float,
    guided_local_search: bool = False,
):
    p = Parser(instance_path)
    instance = p.parse()
    Utils.set_current_instance(instance)

    scheduler = ApexUpgradedScheduler(
        instance_data=instance,
        time_limit_seconds=time_limit,
        population_size=config["population_size"],
        ga_time_fraction=0.45,
        crossover_rate=config["crossover_rate"],
        mutation_rate=config["mutation_rate"],
        tournament_k=config["tournament_k"],
        elite_fraction=config["elite_fraction"],
        destroy_rebuild_rate=config["destroy_rebuild_rate"],
        block_crossover_rate=config["block_crossover_rate"],
        local_search_budget_seconds=local_search_budget,
        guided_local_search=guided_local_search,
        seed=seed,
        lookahead_limit=6,
        density_percentile=25,
        verbose=False,
    )

    t0 = time.time()
    scheduler.generate_solution()
    elapsed = time.time() - t0

    metrics = scheduler.get_run_metrics()
    save_solution_json(scheduler.ga_ls_solution, instance_path, cfg_name, seed, "GA_LS")

    return {
        "ga_score": metrics["ga_score"],
        "ga_segments": metrics["ga_segments"],
        "ga_ls_score": metrics["ga_ls_score"],
        "ga_ls_segments": metrics["ga_ls_segments"],
        "improvement": metrics["improvement"],
        "improvement_pct": metrics["improvement_pct"],
        "score": metrics["ga_score"],
        "segments": metrics["ga_segments"],
        "GA+LS": metrics["ga_ls_score"],
        "ga_time_s": metrics["ga_time_s"],
        "ls_time_s": metrics["ls_time_s"],
        "time_s": round(elapsed, 2),
        "ls_attempts": metrics["ls_attempts"],
        "ls_improvements": metrics["ls_improvements"],
    }


def prepare_csv_row(row: dict) -> dict:
    """Keep CSV output compact while retaining richer metrics in memory."""
    return {
        "config": row.get("config", ""),
        "instance": row.get("instance", ""),
        "seed": row.get("seed", ""),
        "score": row.get("score", row.get("ga_score", "")),
        "segments": row.get("segments", row.get("ga_segments", "")),
        "time_s": row.get("time_s", ""),
        "GA+LS": row.get("GA+LS", row.get("ga_ls_score", row.get("score", ""))),
        "improvement": row.get("improvement", ""),
    }


def csv_header_matches(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return header == CSV_FIELDS


def rewrite_csv_schema(path: Path) -> None:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [prepare_csv_row(row) for row in reader]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_row_csv(path: Path, row: dict, write_header: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not csv_header_matches(path):
        rewrite_csv_schema(path)

    needs_header = write_header or not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(prepare_csv_row(row))


def append_result_csvs(row: dict) -> None:
    """Append one result row to all standard CSV outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    append_row_csv(RESULTS_DIR / "all_results.csv", row, write_header=False)
    append_row_csv(RESULTS_DIR / f"{row['config']}.csv", row, write_header=False)
    append_row_csv(
        RESULTS_DIR / f"{row['instance']}_all_configs.csv",
        row,
        write_header=False,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Eksperimente GA: instanca x seeds per konfigurim.")
    p.add_argument(
        "--configs",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Emrat e konfigurimeve (p.sh. Intensiv Balancuar). Pa argument: te gjitha.",
    )
    p.add_argument(
        "--instances",
        nargs="*",
        default=None,
        metavar="PATH",
        help="Lista e instancave per testim. Pa argument: te gjitha instancat.",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help="Seed-et per ekzekutim. Pa argument: 1..10.",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT,
        help="Buxheti total per ekzekutim ne sekonda (default 300, max 600).",
    )
    p.add_argument(
        "--ls-budget",
        type=float,
        default=LOCAL_SEARCH_BUDGET,
        help="Buxheti i rezervuar per Local Search pas GA ne sekonda.",
    )
    p.add_argument(
        "--guided-ls",
        action="store_true",
        help="Aktivizon Guided Local Search eksperimentale gjate fazes LS.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.configs:
        unknown = [n for n in args.configs if n not in CONFIGS]
        if unknown:
            raise SystemExit(f"Konfigurim i panjohur: {unknown}. Te mundshem: {list(CONFIGS.keys())}")
        configs_to_run = {n: CONFIGS[n] for n in args.configs}
    else:
        configs_to_run = dict(CONFIGS)

    instances_to_run = args.instances if args.instances else list(INSTANCES)
    seeds_to_run = args.seeds if args.seeds else list(SEEDS)
    time_limit = min(max(1.0, float(args.time_limit)), 600.0)
    local_search_budget = max(0.0, float(args.ls_budget))

    total_runs = len(configs_to_run) * len(instances_to_run) * len(seeds_to_run)
    run_idx = 0
    rows = []

    for cfg_name, cfg in configs_to_run.items():
        for inst_path in instances_to_run:
            inst_name = Path(inst_path).stem.replace("_input", "")
            for seed in seeds_to_run:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] {cfg_name} | {inst_name} | seed={seed}",
                    end=" ... ",
                    flush=True,
                )
                try:
                    res = run_single(
                        inst_path,
                        seed,
                        cfg,
                        cfg_name,
                        time_limit,
                        local_search_budget,
                        guided_local_search=args.guided_ls,
                    )
                    print(
                        f"GA={res['ga_score']}  GA+LS={res['ga_ls_score']}  "
                        f"imp={res['improvement']}  time={res['time_s']}s"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
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
                    "instance": inst_name,
                    "seed": seed,
                    "ga_score": res["ga_score"],
                    "ga_segments": res["ga_segments"],
                    "ga_ls_score": res["ga_ls_score"],
                    "ga_ls_segments": res["ga_ls_segments"],
                    "improvement": res["improvement"],
                    "improvement_pct": res["improvement_pct"],
                    "score": res["score"],
                    "segments": res["segments"],
                    "GA+LS": res["GA+LS"],
                    "ga_time_s": res["ga_time_s"],
                    "ls_time_s": res["ls_time_s"],
                    "time_s": res["time_s"],
                    "ls_attempts": res["ls_attempts"],
                    "ls_improvements": res["ls_improvements"],
                }

                rows.append(row)
                append_result_csvs(row)

        print(f"\n  => Completed {cfg_name} (JSON + CSV written)\n")

    print_summary(rows, config_order=list(configs_to_run.keys()))


def print_summary(rows: list[dict], config_order=None):
    from collections import defaultdict

    def as_int(row, key, fallback_key=None):
        value = row.get(key, "")
        if value == "" and fallback_key is not None:
            value = row.get(fallback_key, "")
        return int(float(value)) if value != "" else 0

    for r in rows:
        r["ga_score"] = as_int(r, "ga_score", "score")
        r["ga_ls_score"] = as_int(r, "ga_ls_score", "GA+LS")
        r["improvement"] = as_int(r, "improvement")

    by_cfg_inst_ga = defaultdict(list)
    by_cfg_inst_ls = defaultdict(list)
    by_cfg_inst_imp = defaultdict(list)
    for r in rows:
        key = (r["config"], r["instance"])
        by_cfg_inst_ga[key].append(r["ga_score"])
        by_cfg_inst_ls[key].append(r["ga_ls_score"])
        by_cfg_inst_imp[key].append(r["improvement"])

    instances = []
    seen = set()
    for r in rows:
        if r["instance"] not in seen:
            seen.add(r["instance"])
            instances.append(r["instance"])

    if config_order is not None:
        configs = list(config_order)
    else:
        configs = []
        seen_c = set()
        for r in rows:
            c = r["config"]
            if c not in seen_c:
                seen_c.add(c)
                configs.append(c)

    summary_rows = []

    print("\n" + "=" * 140)
    print("SUMMARY: GA+LS Best / Avg / Worst and Avg improvement per instance per config")
    print("=" * 140)

    header = f"{'Instance':<25}"
    for cfg in configs:
        header += f" | {'Best':>7} {'Avg':>8} {'Worst':>7} {'+LSavg':>7}  [{cfg}]"
    print(header)
    print("-" * len(header))

    for inst in instances:
        line = f"{inst:<25}"
        for cfg in configs:
            scores = by_cfg_inst_ls.get((cfg, inst), [0])
            improvements = by_cfg_inst_imp.get((cfg, inst), [0])
            ga_scores = by_cfg_inst_ga.get((cfg, inst), [0])
            best = max(scores)
            worst = min(scores)
            avg = sum(scores) / len(scores)
            avg_imp = sum(improvements) / len(improvements)
            avg_ga = sum(ga_scores) / len(ga_scores)
            line += f" | {best:>7} {avg:>8.1f} {worst:>7} {avg_imp:>7.1f}"
            summary_rows.append({
                "instance": inst,
                "config": cfg,
                "ga_best": max(ga_scores),
                "ga_avg": round(avg_ga, 1),
                "ga_worst": min(ga_scores),
                "ga_ls_best": best,
                "ga_ls_avg": round(avg, 1),
                "ga_ls_worst": worst,
                "best_gain": best - max(ga_scores),
                "avg_improvement": round(avg_imp, 1),
                "runs": len(scores),
            })
        print(line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance",
            "config",
            "ga_best",
            "ga_avg",
            "ga_worst",
            "ga_ls_best",
            "ga_ls_avg",
            "ga_ls_worst",
            "best_gain",
            "avg_improvement",
            "runs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n  => Summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("BEST CONFIG per instance:")
    print("=" * 60)
    for inst in instances:
        best_cfg = None
        best_avg = -1
        for cfg in configs:
            scores = by_cfg_inst_ls.get((cfg, inst), [0])
            avg = sum(scores) / len(scores)
            if avg > best_avg:
                best_avg = avg
                best_cfg = cfg
        print(f"  {inst:<25} => {best_cfg} (avg={best_avg:.1f})")


if __name__ == "__main__":
    main()
