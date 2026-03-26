import bisect
import random
from typing import List, Optional, Set, Tuple

from models.schedule import Schedule
from models.solution import Solution
from scheduler.beam_search_scheduler import BeamSearchScheduler


class RandomizedScheduler(BeamSearchScheduler):
    """
    Randomized scheduler based on repeated randomized greedy construction.

    It reuses preprocessing, candidate generation and local search utilities from
    BeamSearchScheduler, but explores the solution space stochastically.
    """

    def __init__(
        self,
        instance_data,
        iterations: int = 250,
        candidate_pool: int = 12,
        seed: Optional[int] = None,
        lookahead_limit: int = 4,
        density_percentile: int = 25,
        verbose: bool = True,
    ):
        super().__init__(
            instance_data=instance_data,
            beam_width=1,
            lookahead_limit=lookahead_limit,
            density_percentile=density_percentile,
            verbose=verbose,
        )
        self.iterations = max(1, iterations)
        self.candidate_pool = max(1, candidate_pool)
        self.rng = random.Random(seed)

    def _pick_candidate(
        self,
        candidates: List[Tuple[int, int, int, object, int, int]],
    ) -> Tuple[int, int, int, object, int, int]:
        """Pick one candidate using a weighted random choice from top candidates."""
        closing = self.instance_data.closing_time

        ranked = sorted(
            candidates,
            key=lambda x: x[0] + (closing - x[5]) * self.avg_score_per_min,
            reverse=True,
        )

        pool_size = min(len(ranked), self.candidate_pool + self.rng.randint(0, 3))
        pool = ranked[:pool_size]

        # Small exploitation bias: choose best candidate directly sometimes.
        if self.rng.random() < 0.25:
            return pool[0]

        # Rank-based weights (top gets highest weight).
        weights = list(range(pool_size, 0, -1))
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def _randomized_construction(self) -> Solution:
        opening = self.instance_data.opening_time
        closing = self.instance_data.closing_time

        time = opening
        prev_ch_id = None
        prev_genre = ""
        genre_streak = 0

        scheduled: List[Schedule] = []
        used_progs: Set[str] = set()
        total_score = 0

        guard = 0
        while time < closing and guard < 10000:
            guard += 1

            candidates = self._get_candidates(
                time=time,
                prev_ch_id=prev_ch_id,
                prev_genre=prev_genre,
                genre_streak=genre_streak,
                used_progs=used_progs,
            )

            if not candidates:
                idx = bisect.bisect_right(self.times, time)
                if idx < len(self.times) and self.times[idx] < closing:
                    time = self.times[idx]
                    continue
                break

            # Exploration move: sometimes skip to next decision point.
            if self.rng.random() < 0.08:
                idx = bisect.bisect_right(self.times, time)
                if idx < len(self.times) and self.times[idx] < closing:
                    time = self.times[idx]
                    continue

            seg_score, ch_idx, ch_id, prog, seg_start, seg_end = self._pick_candidate(candidates)

            scheduled.append(
                Schedule(
                    program_id=prog.program_id,
                    channel_id=ch_id,
                    start=seg_start,
                    end=seg_end,
                    fitness=seg_score,
                    unique_program_id=prog.unique_id,
                )
            )

            total_score += seg_score
            used_progs.add(prog.unique_id)

            if prog.genre == prev_genre:
                genre_streak += 1
            else:
                genre_streak = 1
                prev_genre = prog.genre

            prev_ch_id = ch_id
            time = seg_end

        return Solution(scheduled_programs=scheduled, total_score=total_score)

    def generate_solution(self) -> Solution:
        if self.verbose:
            print("\n" + "=" * 70)
            print("RANDOMIZED SCHEDULER")
            print(f"Channels: {self.n_channels}, Iterations: {self.iterations}")
            print("=" * 70 + "\n")

        best_solution = Solution([], 0)

        for _ in range(self.iterations):
            candidate = self._randomized_construction()
            if candidate.total_score > best_solution.total_score:
                best_solution = candidate

        # Improve best random result using local search from parent class.
        iter_limit = 30 if self.n_channels <= 50 else 15
        best_solution = self._local_search(best_solution, max_iter=iter_limit)

        if self.verbose:
            print(f"Best random score: {best_solution.total_score}")
            print(f"Programs selected: {len(best_solution.scheduled_programs)}")
            print()

        return best_solution
