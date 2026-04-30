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
from utils.utils import Utils

_DEFAULT_TIME_LIMIT_SEC = 300.0


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
        help="Time budget in seconds (default 300; max 300 for this build)",
    )
    parser_arg.add_argument(
        "--ga-population",
        type=int,
        default=40,
        help="GA population size (default 40; scales down on large instances)",
    )
    parser_arg.add_argument(
        "--ga-elite",
        type=int,
        default=4,
        help="Elite count per generation (used as fraction of population)",
    )
    parser_arg.add_argument(
        "--ga-tournament",
        type=int,
        default=3,
        help="Tournament size k for parent selection",
    )
    parser_arg.add_argument(
        "--ga-crossover-rate",
        type=float,
        default=0.85,
        help="Crossover probability per offspring",
    )
    parser_arg.add_argument(
        "--ga-mutation-rate",
        type=float,
        default=0.40,
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

    args = parser_arg.parse_args()

    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        file_path = args.input_file
    else:
        file_path = select_file()

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
    tl = min(300.0, tl)

    pop = max(8, args.ga_population)
    print("\nRunning Apex Upgraded Scheduler (structure-aware GA + window repair)")
    scheduler = ApexUpgradedScheduler(
        instance_data=instance,
        time_limit_seconds=tl,
        population_size=pop,
        ga_time_fraction=0.45,
        crossover_rate=args.ga_crossover_rate,
        mutation_rate=args.ga_mutation_rate,
        tournament_k=args.ga_tournament,
        elite_fraction=max(1, args.ga_elite) / pop,
        seed=(int(args.seed) if args.seed is not None else None),
        lookahead_limit=max(4, args.lookahead),
        density_percentile=args.density_percentile,
        verbose=True,
    )

    solution = scheduler.generate_solution()
    print(f"\n[OK] Generated solution with total score: {solution.total_score}")

    algorithm_name = type(scheduler).__name__.lower()
    serializer = SolutionSerializer(input_file_path=file_path, algorithm_name=algorithm_name)
    serializer.serialize(solution)

    print("[OK] Solution saved to output file")
    print(f"Wall time: {time.perf_counter() - wall_perf_t0:.2f}s")


if __name__ == "__main__":
    main()
