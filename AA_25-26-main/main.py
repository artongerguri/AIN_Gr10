from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.randomized_scheduler import RandomizedScheduler
from scheduler.simulated_annealing_scheduler import SimulatedAnnealingScheduler
from utils.utils import Utils
import argparse
import os


def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")
    parser_arg.add_argument("--input", "-i", dest="input_file", help="Path to input JSON (optional)")
    parser_arg.add_argument(
        "--algorithm",
        "-a",
        choices=["beam", "random", "sa"],
        default=None,
        help="Scheduling algorithm to run (if omitted, it will be selected interactively)",
    )
    parser_arg.add_argument(
        "--iterations",
        type=int,
        default=250,
        help="Number of random restarts for random scheduler",
    )
    parser_arg.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible random scheduler runs",
    )
    parser_arg.add_argument(
        "--temperature",
        type=float,
        default=100.0,
        help="Initial temperature for simulated annealing",
    )
    parser_arg.add_argument(
        "--cooling-rate",
        type=float,
        default=0.995,
        help="Cooling rate for simulated annealing (0.8 - 0.99999)",
    )
    parser_arg.add_argument(
        "--sa-steps",
        type=int,
        default=120,
        help="Number of neighborhood moves per SA iteration",
    )
    
    args = parser_arg.parse_args()

    algorithm = args.algorithm
    if algorithm is None:
        print("\nSelect algorithm:")
        options = [("beam", "Beam Search"), ("random", "Randomized"), ("sa", "Simulated Annealing")]
        for idx, (_, label) in enumerate(options, start=1):
            print(f"{idx}. {label}")

        while True:
            choice = input("Choose algorithm (1-3): ").strip()
            if choice in {"1", "2", "3"}:
                algorithm = options[int(choice) - 1][0]
                break
            print("Invalid choice, please enter 1, 2, or 3.")
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

    # Shared default optimized parameters
    lookahead = 4
    percentile = 25

    if algorithm == "beam":
        print("\nRunning Beam Search Scheduler")
        scheduler = BeamSearchScheduler(
            instance_data=instance,
            beam_width=100,
            lookahead_limit=lookahead,
            density_percentile=percentile,
            verbose=False,
        )
    elif algorithm == "random":
        print("\nRunning Randomized Scheduler")
        scheduler = RandomizedScheduler(
            instance_data=instance,
            iterations=args.iterations,
            seed=args.seed,
            lookahead_limit=lookahead,
            density_percentile=percentile,
            verbose=False,
        )
    else:
        print("\nRunning Simulated Annealing Scheduler")
        scheduler = SimulatedAnnealingScheduler(
            instance_data=instance,
            iterations=args.iterations,
            initial_temperature=args.temperature,
            cooling_rate=args.cooling_rate,
            steps_per_iteration=args.sa_steps,
            seed=args.seed,
            lookahead_limit=lookahead,
            density_percentile=percentile,
            verbose=False,
        )

    solution = scheduler.generate_solution()
    print(f"\n✓ Generated solution with total score: {solution.total_score}")

    algorithm_name = type(scheduler).__name__.lower()
    serializer = SolutionSerializer(input_file_path=file_path, algorithm_name=algorithm_name)
    serializer.serialize(solution)

    print(f"✓ Solution saved to output file")


if __name__ == "__main__":
    main()
