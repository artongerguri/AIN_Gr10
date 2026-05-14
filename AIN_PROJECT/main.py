"""
Entry point: Apex Upgraded scheduler only (TV scheduling).
"""
import argparse
import os
import time

from parser.file_selector import select_file
from parser.parser import Parser
from scheduler.apex_upgraded_scheduler import ApexUpgradedScheduler
from serializer.serializer import SolutionSerializer
from utils.debug_breakpoints import debug_breakpoint, enable_debug_breakpoints
from utils.utils import Utils

_DEFAULT_TIME_LIMIT_SEC = 300.0
_MAX_TIME_LIMIT_SEC = 600.0


def main():
    wall_perf_t0 = time.perf_counter()
    parser_arg = argparse.ArgumentParser(
        description="TV scheduling — Apex Upgraded (block GA + window repair)"
    )
    parser_arg.add_argument(
        "--input", "-i", dest="input_file", help="Path to input JSON (optional)"
    )
    parser_arg.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs",
    )
    parser_arg.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time budget in seconds (default 300; max 600 for this build)",
    )
    parser_arg.add_argument(
        "--ls-budget",
        type=float,
        default=30.0,
        help="Seconds reserved for Local Search after GA",
    )
    parser_arg.add_argument(
        "--ga-population",
        type=int,
        default=60,
        help="GA population size (default 60; scales down on large instances)",
    )
    parser_arg.add_argument(
        "--ga-elite",
        type=int,
        default=12,
        help="Elite count per generation (used as fraction of population)",
    )
    parser_arg.add_argument(
        "--ga-tournament",
        type=int,
        default=5,
        help="Tournament size k for parent selection",
    )
    parser_arg.add_argument(
        "--ga-crossover-rate",
        type=float,
        default=0.55,
        help="Crossover probability per offspring",
    )
    parser_arg.add_argument(
        "--ga-mutation-rate",
        type=float,
        default=0.50,
        help="Mutation probability per offspring",
    )
    parser_arg.add_argument(
        "--lookahead",
        type=int,
        default=6,
        help="Lookahead limit for candidate generation",
    )
    parser_arg.add_argument(
        "--density-percentile",
        type=int,
        default=25,
        help="Top percentile for avg score/min heuristic (1–100)",
    )
    parser_arg.add_argument(
        "--debug-breakpoints",
        action="store_true",
        help="Stop once at the main algorithm methods for step-by-step debugging",
    )
    parser_arg.add_argument(
        "--debug-breakpoints-repeat",
        action="store_true",
        help="Stop every time a debug breakpoint is reached",
    )
    parser_arg.add_argument(
        "--debug-breakpoint-labels",
        default=None,
        help="Comma-separated breakpoint labels to enable, e.g. ApexScheduler._ga,ApexUpgradedScheduler._local_search",
    )

    args = parser_arg.parse_args()

    if args.debug_breakpoints:
        enable_debug_breakpoints(
            repeat=args.debug_breakpoints_repeat,
            labels=args.debug_breakpoint_labels,
        )

    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        file_path = args.input_file
    else:
        file_path = select_file()

    debug_breakpoint(
        "main.before_parse",
        input_file=file_path,
        seed=args.seed,
        time_limit=args.time_limit,
        ls_budget=args.ls_budget,
    )
    parser = Parser(file_path)
    instance = parser.parse()
    Utils.set_current_instance(instance)

    print("\nOpening time:", instance.opening_time)
    print("Closing time:", instance.closing_time)
    print(f"Total Channels: {len(instance.channels)}")

    tl = (
        max(1.0, float(args.time_limit))
        if args.time_limit is not None
        else _DEFAULT_TIME_LIMIT_SEC
    )
    tl = min(_MAX_TIME_LIMIT_SEC, tl)

    pop = max(8, args.ga_population)
    print("\nRunning Apex Upgraded Scheduler (structure-aware GA + window repair)")
    debug_breakpoint(
        "main.before_scheduler",
        channels=len(instance.channels),
        opening_time=instance.opening_time,
        closing_time=instance.closing_time,
        population=pop,
    )
    scheduler = ApexUpgradedScheduler(
        instance_data=instance,
        time_limit_seconds=tl,
        population_size=pop,
        ga_time_fraction=0.45,
        crossover_rate=args.ga_crossover_rate,
        mutation_rate=args.ga_mutation_rate,
        tournament_k=args.ga_tournament,
        elite_fraction=max(1, args.ga_elite) / pop,
        local_search_budget_seconds=max(0.0, args.ls_budget),
        seed=(int(args.seed) if args.seed is not None else None),
        lookahead_limit=max(4, args.lookahead),
        density_percentile=args.density_percentile,
        verbose=True,
    )

    debug_breakpoint("main.before_generate_solution", scheduler=type(scheduler).__name__)
    scheduler.generate_solution()
    metrics = scheduler.get_run_metrics()
    if scheduler.ga_ls_solution is not None:
        SolutionSerializer(file_path, "apexupgradedscheduler").serialize(
            scheduler.ga_ls_solution
        )
    debug_breakpoint(
        "main.after_generate_solution",
        ga_score=metrics["ga_score"],
        ga_ls_score=metrics["ga_ls_score"],
        improvement=metrics["improvement"],
    )
    print(
        f"\n[OK] GA result: score={metrics['ga_score']} "
        f"segments={metrics['ga_segments']}"
    )
    print(
        f"[OK] GA+LS result: score={metrics['ga_ls_score']} "
        f"segments={metrics['ga_ls_segments']} "
        f"improvement={metrics['improvement']} "
        f"({metrics['improvement_pct']}%)"
    )

    print(f"Wall time: {time.perf_counter() - wall_perf_t0:.2f}s")


if __name__ == "__main__":
    main()
