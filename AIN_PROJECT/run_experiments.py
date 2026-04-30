"""
Eksperimente: 4 konfigurime x 17 instanca x 10 seeds.
  python run_experiments.py
  python run_experiments.py --configs Intensiv Balancuar

Per cdo ekzekutim:
  - Ruan zgjidhjen JSON ne data/output/ (si SolutionSerializer)
  - Ruan rreshtin ne results/all_results.csv  +  results/<config>.csv
  - Ne fund printon summary + ruan results/summary.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path

from parser.parser import Parser
from scheduler.apex_upgraded_scheduler import ApexUpgradedScheduler
from serializer.serializer import SolutionSerializer
from utils.utils import Utils

INSTANCES = [
    "data/input/toy.json",
    "data/input/canada_pw.json",
    "data/input/usa_tv_input.json",
    "data/input/uk_tv_input.json",
    "data/input/croatia_tv_input.json",
    "data/input/germany_tv_input.json",
    "data/input/kosovo_tv_input.json",
    "data/input/china_pw.json",
    "data/input/youtube_premium.json",
    "data/input/youtube_gold.json",
    "data/input/us_iptv.json",
    "data/input/uk_iptv.json",
    "data/input/spain_iptv.json",
    "data/input/france_iptv.json",
    "data/input/australia_iptv.json",
    "data/input/singapore_pw.json",
    "data/input/netherlands_tv_input.json",
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

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("data/output")

CSV_FIELDS = ["config", "instance", "seed", "score", "segments", "time_s"]


def save_solution_json(solution, instance_path: str, cfg_name: str, seed: int):
    """Ruan zgjidhjen si JSON ne data/output/ me emrin e formatuar."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(instance_path).stem.replace("_input", "")
    score = int(solution.total_score)
    fname = f"{base}_output_{cfg_name}_seed{seed}_{score}.json"
    out_path = OUTPUT_DIR / fname

    schedules = []
    for s in solution.scheduled_programs:
        schedules.append({
            "program_id": s.program_id,
            "channel_id": s.channel_id,
            "start": s.start,
            "end": s.end,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scheduled_programs": schedules}, f, indent=4, ensure_ascii=False)


def run_single(instance_path: str, seed: int, config: dict, cfg_name: str):
    p = Parser(instance_path)
    instance = p.parse()
    Utils.set_current_instance(instance)

    scheduler = ApexUpgradedScheduler(
        instance_data=instance,
        time_limit_seconds=TIME_LIMIT,
        population_size=config["population_size"],
        ga_time_fraction=0.45,
        crossover_rate=config["crossover_rate"],
        mutation_rate=config["mutation_rate"],
        tournament_k=config["tournament_k"],
        elite_fraction=config["elite_fraction"],
        destroy_rebuild_rate=config["destroy_rebuild_rate"],
        block_crossover_rate=config["block_crossover_rate"],
        seed=seed,
        lookahead_limit=6,
        density_percentile=25,
        verbose=False,
    )

    t0 = time.time()
    solution = scheduler.generate_solution()
    elapsed = time.time() - t0

    save_solution_json(solution, instance_path, cfg_name, seed)

    return {
        "score": solution.total_score,
        "segments": len(solution.scheduled_programs),
        "time_s": round(elapsed, 2),
    }


def append_row_csv(path: Path, row: dict, write_header: bool):
    """Shton nje rresht ne CSV (krijon file nese nuk ekziston)."""
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description="Eksperimente GA: instanca x seeds per konfigurim.")
    p.add_argument(
        "--configs",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Emrat e konfigurimeve (p.sh. Intensiv Balancuar). Pa argument: te gjitha.",
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_csv = RESULTS_DIR / "all_results.csv"
    first_all = True

    total_runs = len(configs_to_run) * len(INSTANCES) * len(SEEDS)
    run_idx = 0

    for cfg_name, cfg in configs_to_run.items():
        cfg_csv = RESULTS_DIR / f"{cfg_name}.csv"
        first_cfg = True

        for inst_path in INSTANCES:
            inst_name = Path(inst_path).stem.replace("_input", "")
            for seed in SEEDS:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] {cfg_name} | {inst_name} | seed={seed}",
                    end=" ... ",
                    flush=True,
                )
                try:
                    res = run_single(inst_path, seed, cfg, cfg_name)
                    print(f"score={res['score']}  time={res['time_s']}s")
                except Exception as e:
                    print(f"ERROR: {e}")
                    res = {"score": 0, "segments": 0, "time_s": 0}

                row = {
                    "config": cfg_name,
                    "instance": inst_name,
                    "seed": seed,
                    "score": res["score"],
                    "segments": res["segments"],
                    "time_s": res["time_s"],
                }

                append_row_csv(cfg_csv, row, write_header=first_cfg)
                first_cfg = False
                append_row_csv(all_csv, row, write_header=first_all)
                first_all = False

        print(f"\n  => Saved {cfg_csv}\n")

    print_summary(all_csv, config_order=list(configs_to_run.keys()))


def print_summary(all_csv_path: Path, config_order=None):
    from collections import defaultdict

    rows = []
    with open(all_csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["score"] = int(r["score"])
            rows.append(r)

    by_cfg_inst = defaultdict(list)
    for r in rows:
        by_cfg_inst[(r["config"], r["instance"])].append(r["score"])

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

    summary_path = RESULTS_DIR / "summary.csv"
    summary_rows = []

    print("\n" + "=" * 120)
    print("SUMMARY: Best / Avg / Worst score per instance per config")
    print("=" * 120)

    header = f"{'Instance':<25}"
    for cfg in configs:
        header += f" | {'Best':>7} {'Avg':>8} {'Worst':>7}  [{cfg}]"
    print(header)
    print("-" * len(header))

    for inst in instances:
        line = f"{inst:<25}"
        for cfg in configs:
            scores = by_cfg_inst.get((cfg, inst), [0])
            best = max(scores)
            worst = min(scores)
            avg = sum(scores) / len(scores)
            line += f" | {best:>7} {avg:>8.1f} {worst:>7}"
            summary_rows.append({
                "instance": inst,
                "config": cfg,
                "best": best,
                "avg": round(avg, 1),
                "worst": worst,
                "runs": len(scores),
            })
        print(line)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "config", "best", "avg", "worst", "runs"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n  => Summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("BEST CONFIG per instance:")
    print("=" * 60)
    for inst in instances:
        best_cfg = None
        best_avg = -1
        for cfg in configs:
            scores = by_cfg_inst.get((cfg, inst), [0])
            avg = sum(scores) / len(scores)
            if avg > best_avg:
                best_avg = avg
                best_cfg = cfg
        print(f"  {inst:<25} => {best_cfg} (avg={best_avg:.1f})")


if __name__ == "__main__":
    main()
