import bisect
import math
import random
from typing import List, Optional, Set, Tuple, Dict

from models.schedule import Schedule
from models.solution import Solution
from scheduler.beam_search_scheduler import BeamSearchScheduler

class ImprovedAnnealingScheduler(BeamSearchScheduler):
    """
    Një version i përmirësuar që përdor Hybrid Large Neighborhood Search (LNS) 
    dhe Simulated Annealing.
    """

    def __init__(
        self,
        instance_data,
        iterations: int = 300,
        initial_temperature: float = 150.0,
        cooling_rate: float = 0.992,
        candidate_pool: int = 16,
        steps_per_iteration: int = 150,
        seed: Optional[int] = None,
        lookahead_limit: int = 5,
        density_percentile: int = 20,
        verbose: bool = True,
    ):
        super().__init__(
            instance_data=instance_data,
            beam_width=1,
            lookahead_limit=lookahead_limit,
            density_percentile=density_percentile,
            verbose=verbose,
        )
        self.iterations = iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.candidate_pool = candidate_pool
        self.steps_per_iteration = steps_per_iteration
        self.rng = random.Random(seed)
        
        # Heuristika për vlerësimin e densitetit të pikëve
        self.avg_score_per_min = self._estimate_avg_efficiency()

    def _estimate_avg_efficiency(self) -> float:
        """Llogarit një mesatare të pikëve për minutë për të udhëhequr zgjedhjet."""
        sample_progs = list(self.prog_by_id.values())[:100]
        if not sample_progs: return 1.0
        efficiencies = [p[0].score / max(1, p[0].end - p[0].start) for p in sample_progs]
        return sum(efficiencies) / len(efficiencies)

    def _energy(self, solution: Solution) -> float:
        """
        Funksioni i energjisë: Prioritet ka Score, 
        pastaj dënimi për kohën e papërdorur (closing_time - last_end).
        """
        if not solution.scheduled_programs:
            return 1e9
        
        last_end = solution.scheduled_programs[-1].end
        gap_penalty = (self.instance_data.closing_time - last_end) * 0.1
        return -(solution.total_score - gap_penalty)

    def _get_neighbor(self, current_sol: Solution) -> Solution:
        """
        Krijon një fqinj duke përdorur një kombinim të:
        1. Prerjes rastësore (Tail Cut)
        2. Heqjes së segmenteve me vlerë të ulët (Targeted Removal)
        """
        progs = current_sol.scheduled_programs
        if not progs:
            return self._regrow_from_prefix([])

        rand_val = self.rng.random()
        
        # Strategjia 1: Prerje e shkurtër në fund (60% e kohës)
        if rand_val < 0.60:
            cut_idx = self.rng.randint(int(len(progs) * 0.6), len(progs))
            return self._regrow_from_prefix(progs[:cut_idx])
        
        # Strategjia 2: Prerje e thellë (30% e kohës)
        elif rand_val < 0.90:
            cut_idx = self.rng.randint(0, len(progs) // 2 + 1)
            return self._regrow_from_prefix(progs[:cut_idx])
            
        # Strategjia 3: "Mutation" - Heqim një program të dobët në mes (10%)
        else:
            # Gjejmë programin me fitness më të ulët
            min_fit_idx = 0
            for i in range(len(progs)):
                if progs[i].fitness < progs[min_fit_idx].fitness:
                    min_fit_idx = i
            return self._regrow_from_prefix(progs[:min_fit_idx])

    def _regrow_from_prefix(self, prefix: List[Schedule]) -> Solution:
        """Version më i shpejtë dhe i optimizuar i ndërtimit të zgjidhjes."""
        time, prev_ch_id, prev_genre, genre_streak, used_progs = self._state_from_prefix(prefix)
        closing = self.instance_data.closing_time
        
        scheduled = prefix[:]
        total_score = sum(s.fitness for s in prefix)
        
        while time < closing:
            candidates = self._get_candidates(
                time=time,
                prev_ch_id=prev_ch_id,
                prev_genre=prev_genre,
                genre_streak=genre_streak,
                used_progs=used_progs,
            )

            if not candidates:
                # Skip to next available time slot
                idx = bisect.bisect_right(self.times, time)
                if idx < len(self.times) and self.times[idx] < closing:
                    time = self.times[idx]
                    continue
                break

            # Renditja sipas densitetit të pikëve dhe penalitetit të kohës
            # Formula: Score / Duration + (Bonus nëse plotëson kohën mirë)
            pool = sorted(
                candidates,
                key=lambda x: (x[0] / max(1, x[5]-x[4])) + (0.1 if x[5] <= closing else -100),
                reverse=True
            )[:self.candidate_pool]

            # Zgjedhje probabilistike (Weighted Choice)
            if self.rng.random() < 0.7:
                pick = pool[0] # Greedy
            else:
                pick = self.rng.choice(pool) # Explore

            seg_score, _, ch_id, prog, seg_start, seg_end = pick
            
            scheduled.append(Schedule(
                program_id=prog.program_id,
                channel_id=ch_id,
                start=seg_start,
                end=seg_end,
                fitness=seg_score,
                unique_program_id=prog.unique_id
            ))
            
            total_score += seg_score
            used_progs.add(prog.unique_id)
            
            genre_streak = (genre_streak + 1) if prog.genre == prev_genre else 1
            prev_genre = prog.genre
            prev_ch_id = ch_id
            time = seg_end

        return Solution(scheduled_programs=scheduled, total_score=total_score)

    def _anneal(self, initial_sol: Solution) -> Solution:
        current_sol = initial_sol
        best_local_sol = initial_sol
        
        temp = self.initial_temperature
        
        for step in range(self.steps_per_iteration):
            neighbor = self._get_neighbor(current_sol)
            
            delta = self._energy(neighbor) - self._energy(current_sol)
            
            # Pranimi i lëvizjes (Metropolis Criterion)
            if delta < 0 or (temp > 0 and self.rng.random() < math.exp(-delta / temp)):
                current_sol = neighbor
                if current_sol.total_score > best_local_sol.total_score:
                    best_local_sol = current_sol
            
            temp *= self.cooling_rate
            
        return best_local_sol

    def generate_solution(self) -> Solution:
        if self.verbose:
            print(f"Starting Improved SA: Iter={self.iterations}, Steps={self.steps_per_iteration}")

        best_global_sol = Solution([], 0)
        
        # Adaptive Iterations based on problem size
        actual_iters = self.iterations
        if self.n_channels > 60:
            actual_iters = min(actual_iters, 50) # Për shpejtësi në instanca të mëdha

        for i in range(actual_iters):
            # Fillojmë nga një pikë nisjeje e ndryshme çdo herë (Multi-start)
            initial = self._regrow_from_prefix([])
            candidate = self._anneal(initial)
            
            if candidate.total_score > best_global_sol.total_score:
                best_global_sol = candidate
                if self.verbose:
                    print(f" Iter {i}: New Best Score = {best_global_sol.total_score}")

        return best_global_sol

    def _state_from_prefix(self, prefix: List[Schedule]):
        if not prefix:
            return self.instance_data.opening_time, None, "", 0, set()
        
        last = prefix[-1]
        used = {s.unique_program_id for s in prefix}
        
        # Gjejmë streak-un e zhanrit
        last_info = self.prog_by_id.get(last.unique_program_id)
        prev_genre = last_info[0].genre if last_info else ""
        streak = 0
        for s in reversed(prefix):
            info = self.prog_by_id.get(s.unique_program_id)
            if info and info[0].genre == prev_genre:
                streak += 1
            else:
                break
                
        return last.end, last.channel_id, prev_genre, streak, used


# Backward-compatible name expected by main.py and existing callers.
class SimulatedAnnealingScheduler(ImprovedAnnealingScheduler):
    pass