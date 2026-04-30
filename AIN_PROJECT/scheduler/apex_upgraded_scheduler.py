"""
ApexUpgradedScheduler — Apex-grade seeding + GA with stronger local search,
structure-preserving crossover, and region destroy-rebuild mutations.

Inherits construction, ranking, seeding, and repair from ApexScheduler without
modifying that module. All upgrades live in this file only.
"""

from __future__ import annotations

import bisect
import time as _time
from typing import Dict, List, Optional, Set, Tuple

from models.schedule import Schedule
from models.solution import Solution
from scheduler.apex_scheduler import ApexScheduler


class ApexUpgradedScheduler(ApexScheduler):
    """
    Same 5-minute wall budget as Apex, with:
    - Block / chain identification for crossover and mutations
    - Bounded beam re-optimization inside removed time windows
    - Region-first mutations (weak / switch-heavy / pref-adjacent / random)
    - Elite and final polishing passes when time remains
    """

    _FINAL_RESERVE_S = 4.0

    def __init__(
        self,
        instance_data,
        time_limit_seconds: float = 300.0,
        population_size: int = 60,
        ga_time_fraction: float = 0.45,
        crossover_rate: Optional[float] = None,
        mutation_rate: Optional[float] = None,
        tournament_k: Optional[int] = None,
        elite_fraction: float = 0.20,
        stale_limit: Optional[int] = None,
        destroy_rebuild_rate: float = 0.78,
        block_crossover_rate: float = 0.88,
        polish_rounds: Optional[int] = None,
        seed: Optional[int] = None,
        lookahead_limit: int = 6,
        density_percentile: int = 25,
        verbose: bool = True,
    ):
        tl = min(float(time_limit_seconds), 300.0)
        super().__init__(
            instance_data=instance_data,
            time_limit_seconds=tl,
            population_size=population_size,
            ga_time_fraction=ga_time_fraction,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_k=tournament_k,
            elite_fraction=elite_fraction,
            stale_limit=stale_limit,
            seed=seed,
            lookahead_limit=lookahead_limit,
            density_percentile=density_percentile,
            verbose=verbose,
        )
        self.time_limit = tl
        self._deadline: float = 0.0
        self.destroy_rebuild_rate = destroy_rebuild_rate
        self.block_crossover_rate = block_crossover_rate
        self.polish_rounds = polish_rounds

    def _deadline_ok(self, margin: float = 0.0) -> bool:
        return _time.time() + margin < self._deadline

    # ------------------------------------------------------------------ blocks
    def _is_chain_link(self, a: Schedule, b: Schedule) -> bool:
        if a.channel_id != b.channel_id or a.end != b.start:
            return False
        pi = self.prog_by_id.get(a.unique_program_id)
        if not pi:
            return False
        nxi = self._nxt.get(pi[0].unique_id)
        return nxi is not None and nxi[0].unique_id == b.unique_program_id

    def _identify_blocks(
        self, sched: List[Schedule]
    ) -> List[Tuple[int, int, str]]:
        """
        Maximal same-channel continuation chains become one block.
        Returns (lo, hi, 'chain'|'single') inclusive indices.
        """
        if not sched:
            return []
        blocks: List[Tuple[int, int, str]] = []
        lo = 0
        for i in range(len(sched) - 1):
            if self._is_chain_link(sched[i], sched[i + 1]):
                continue
            kind = "chain" if i > lo else "single"
            blocks.append((lo, i, kind))
            lo = i + 1
        kind = "chain" if len(sched) - 1 > lo else "single"
        blocks.append((lo, len(sched) - 1, kind))
        return blocks

    def _natural_cut_times(self) -> List[int]:
        ts = {self._open, self._close}
        for pb in self.instance_data.priority_blocks:
            ts.add(pb.start)
            ts.add(pb.end)
        return sorted(ts)

    # ---------------------------------------------------------- adaptive params
    def _adaptive_window_params(self) -> Dict[str, int]:
        """Bounded repair sizes from instance scale and remaining wall time."""
        left = max(0.0, self._deadline - _time.time())
        ch = self.n_channels
        if ch > 5000:
            max_segs, beam_w, max_exp = 2, 2, 40
        elif ch > 2000:
            max_segs, beam_w, max_exp = 3, 3, 70
        elif ch > 500:
            max_segs, beam_w, max_exp = 4, 4, 120
        elif ch > 100:
            max_segs, beam_w, max_exp = 5, 5, 200
        else:
            max_segs, beam_w, max_exp = 6, 6, 320

        if left < 25:
            max_segs = max(2, max_segs - 2)
            beam_w = max(2, beam_w - 1)
            max_exp = max(30, max_exp // 2)
        elif left < 60:
            max_exp = int(max_exp * 0.75)

        return {
            "max_segs": max_segs,
            "beam_w": beam_w,
            "max_expansions": max_exp,
            "cand_cap": 10 if ch > 500 else 14,
        }

    # -------------------------------------------------------- repair window pick
    def _select_repair_window(
        self, progs: List[Schedule], rng
    ) -> Tuple[int, int]:
        """Inclusive [lo, hi] segment indices to remove and rebuild."""
        n = len(progs)
        if n <= 1:
            return 0, 0
        p = self._adaptive_window_params()
        w = min(p["max_segs"], max(2, n // 4 + 1))
        w = min(w, n)

        mode = rng.random()
        if mode < 0.28:
            lo = super()._weak_region(progs)
            hi = min(n - 1, lo + w - 1)
        elif mode < 0.52:
            lo = self._switch_heavy_region(progs, w)
            hi = min(n - 1, lo + w - 1)
        elif mode < 0.72 and self._pw:
            lo = self._pref_adjacent_region(progs, w, rng)
            hi = min(n - 1, lo + w - 1)
        else:
            lo = rng.randint(0, n - w)
            hi = lo + w - 1

        lo = max(0, min(lo, n - 1))
        hi = max(lo, min(hi, n - 1))

        blocks = self._identify_blocks(progs)
        for blo, bhi, kind in blocks:
            if kind == "chain" and blo <= lo <= hi <= bhi:
                if rng.random() < 0.72 and hi - lo + 1 < bhi - blo + 1:
                    if rng.random() < 0.5:
                        lo = max(lo, blo)
                        hi = min(hi, blo + min(w, 2) - 1)
        return lo, hi

    def _weak_region(self, progs: List[Schedule], window: int = 3) -> int:
        w = min(window, len(progs))
        worst_avg = float("inf")
        worst_idx = 0
        for i in range(len(progs) - w + 1):
            avg = sum(progs[j].fitness for j in range(i, i + w)) / w
            if avg < worst_avg:
                worst_avg = avg
                worst_idx = i
        return worst_idx

    def _switch_heavy_region(self, progs: List[Schedule], w: int) -> int:
        best_i = 0
        best_sw = -1
        for i in range(len(progs) - w + 1):
            sw = 0
            for j in range(i, i + w - 1):
                if progs[j].channel_id != progs[j + 1].channel_id:
                    sw += 1
            if sw > best_sw:
                best_sw = sw
                best_i = i
        return best_i if best_sw > 0 else self.rng.randint(0, max(0, len(progs) - w))

    def _pref_adjacent_region(
        self, progs: List[Schedule], w: int, rng
    ) -> int:
        """Region around a segment that could overlap a preference window."""
        best_i = 0
        best_gap = 10**9
        for i, s in enumerate(progs):
            pi = self.prog_by_id.get(s.unique_program_id)
            if not pi:
                continue
            g = pi[0].genre
            if g not in self._pref_genres:
                continue
            idx = bisect.bisect_right(self._pw_starts, s.start) - 1
            for k in range(max(0, idx), min(len(self._pw), idx + 4)):
                pw = self._pw[k]
                if pw[2] != g:
                    continue
                gap = min(abs(s.start - pw[0]), abs(s.end - pw[1]))
                if gap < best_gap:
                    best_gap = gap
                    best_i = i
        lo = max(0, best_i - w // 2)
        return lo

    # ----------------------------------------------------------- beam gap fill
    def _beam_fill_gap(
        self,
        t0: int,
        t1: int,
        prev_ch: Optional[int],
        prev_genre: str,
        gs: int,
        frozen_used: Set[str],
        params: Dict[str, int],
        outer_deadline: float,
    ) -> Tuple[List[Schedule], int]:
        """
        Layered bounded beam from t0; each scheduled segment must end <= t1.
        Returns (new_segments, sum of segment scores in the gap).
        """
        beam_w = params["beam_w"]
        max_exp = params["max_expansions"]
        cand_cap = params["cand_cap"]

        # (score, time, prev_ch, prev_genre, streak, [Schedule...], used_set)
        layer: List[Tuple[int, int, Optional[int], str, int, List[Schedule], Set[str]]] = [
            (0, t0, prev_ch, prev_genre, gs, [], set(frozen_used))
        ]
        best_score = 0
        best_segs: List[Schedule] = []
        expansions = 0
        max_layers = min(48, max(4, (t1 - t0) // max(self.min_d, 1) + 2))

        for _ in range(max_layers):
            if not layer or expansions >= max_exp or _time.time() >= outer_deadline:
                break
            nxt: List[Tuple[int, int, Optional[int], str, int, List[Schedule], Set[str]]] = []
            for score, time, pch, pgr, gst, segs, used in layer:
                expansions += 1
                if expansions >= max_exp or _time.time() >= outer_deadline:
                    break

                if time >= t1 or time >= self._close:
                    if score > best_score:
                        best_score = score
                        best_segs = segs[:]
                    continue

                cands = self._gen(time, pch, pgr, gst, used)
                if not cands:
                    idx = bisect.bisect_right(self.times, time)
                    if idx < len(self.times) and self.times[idx] < min(t1, self._close):
                        nxt.append(
                            (score, self.times[idx], pch, pgr, gst, segs, used)
                        )
                    else:
                        if score > best_score:
                            best_score = score
                            best_segs = segs[:]
                    continue

                cands.sort(key=lambda x: x[0], reverse=True)
                br: List[Tuple[int, int, Optional[int], str, int, List[Schedule], Set[str]]] = []
                for c in cands[:cand_cap]:
                    seg_sc, ch_idx, ch_id, prog, ss, se = c
                    if se > t1 or se - ss < self.min_d:
                        continue
                    if prog.unique_id in used:
                        continue
                    ns = gst + 1 if prog.genre == pgr else 1
                    if ns > self._mc:
                        continue
                    if not self._channel_allowed(ch_idx, ss, se):
                        continue
                    nu = set(used)
                    nu.add(prog.unique_id)
                    nsch = segs + [
                        Schedule(
                            prog.program_id,
                            ch_id,
                            ss,
                            se,
                            seg_sc,
                            prog.unique_id,
                        )
                    ]
                    br.append((score + seg_sc, se, ch_id, prog.genre, ns, nsch, nu))

                br.sort(
                    key=lambda st: st[0]
                    + (min(t1, self._close) - st[1]) * self.avg_score_per_min * 0.018,
                    reverse=True,
                )
                for st in br[:beam_w]:
                    nxt.append(st)

            if expansions >= max_exp or _time.time() >= outer_deadline:
                break

            if not nxt:
                break
            nxt.sort(
                key=lambda st: st[0]
                + (min(t1, self._close) - st[1]) * self.avg_score_per_min * 0.018,
                reverse=True,
            )
            cap = max(beam_w * 6, beam_w)
            layer = nxt[:cap]

        for score, _, _, _, _, segs, _ in layer:
            if score > best_score:
                best_score = score
                best_segs = segs
        return best_segs, best_score

    def _window_repair(self, sol: Solution, lo: int, hi: int) -> Solution:
        """Remove segments [lo..hi] and re-optimize the gap with bounded beam."""
        progs = sol.scheduled_programs
        if not progs or lo > hi or lo < 0 or hi >= len(progs):
            return sol
        if not self._deadline_ok(margin=0.05):
            return sol

        params = self._adaptive_window_params()
        span = hi - lo + 1
        if span > params["max_segs"]:
            mid = (lo + hi) // 2
            half = params["max_segs"] // 2
            lo = max(0, mid - half)
            hi = min(len(progs) - 1, lo + params["max_segs"] - 1)

        prefix = progs[:lo]
        suffix = progs[hi + 1 :]
        t0 = prefix[-1].end if prefix else self._open
        t1 = suffix[0].start if suffix else self._close

        if t1 <= t0:
            merged = prefix + suffix
            return self._repair(merged)

        used: Set[str] = set()
        for s in prefix:
            used.add(s.unique_program_id)
        for s in suffix:
            used.add(s.unique_program_id)

        prev_ch: Optional[int] = None
        prev_genre = ""
        gs = 0
        if prefix:
            last = prefix[-1]
            prev_ch = last.channel_id
            pi = self.prog_by_id.get(last.unique_program_id)
            prev_genre = pi[0].genre if pi else ""
            gs = self._streak(prefix)

        outer_deadline = min(self._deadline - 0.08, _time.time() + max(0.4, (t1 - t0) / 400.0 + 0.5))
        mid, _mid_sc = self._beam_fill_gap(
            t0, t1, prev_ch, prev_genre, gs, used, params, outer_deadline
        )
        if not mid and not suffix:
            tail = self._construct(strategy="density", alpha=0.05, prefix=prefix if prefix else None)
            return tail

        raw = prefix + mid + suffix
        return self._repair(raw)

    # -------------------------------------------------------------- mutations
    def _destroy_rebuild_mutation(self, sol: Solution) -> Solution:
        progs = sol.scheduled_programs
        if len(progs) < 2:
            strat = self.rng.choice(["density", "chain", "balanced"])
            return self._construct(strategy=strat, alpha=0.15, extend=self.rng.random() < 0.5)

        lo, hi = self._select_repair_window(progs, self.rng)
        return self._window_repair(sol, lo, hi)

    def _mutate(self, sol: Solution) -> Solution:
        if len(sol.scheduled_programs) < 2:
            return super()._mutate(sol)
        if self.rng.random() < self.destroy_rebuild_rate:
            return self._destroy_rebuild_mutation(sol)
        return super()._mutate(sol)

    # --------------------------------------------------------------- crossover
    def _block_crossover(self, p1: Solution, p2: Solution) -> Solution:
        s1 = p1.scheduled_programs
        s2 = p2.scheduled_programs
        if len(s1) < 2 or len(s2) < 2:
            return p1 if p1.total_score >= p2.total_score else p2

        blocks = self._identify_blocks(s1)
        naturals = self._natural_cut_times()

        if len(blocks) >= 2 and self.rng.random() < 0.62:
            bi = self.rng.randint(1, len(blocks) - 1)
            lo = blocks[bi][0]
            prefix = list(s1[:lo])
        else:
            end_times = sorted({s.end for s in s1} | set(naturals))
            end_times = [t for t in end_times if self._open < t < self._close]
            if len(end_times) < 2:
                return super()._crossover(p1, p2)
            cut_time = self.rng.choice(end_times)
            prefix = [s for s in s1 if s.end <= cut_time]
            if not prefix or len(prefix) >= len(s1):
                prefix = s1[: max(1, len(s1) // 2)]

        cut_time = prefix[-1].end if prefix else self._open
        used = {s.unique_program_id for s in prefix}

        middle: List[Schedule] = []
        for s in s2:
            if s.unique_program_id in used:
                continue
            if s.start < cut_time:
                continue
            ch_idx = None
            pi = self.prog_by_id.get(s.unique_program_id)
            if pi:
                ch_idx = pi[1]
            if ch_idx is not None and not self._channel_allowed(ch_idx, s.start, s.end):
                continue
            middle.append(s)

        raw = prefix + middle
        child = self._repair(raw)

        if self._deadline_ok(margin=1.0) and self.rng.random() < 0.35:
            pp = child.scheduled_programs
            if len(pp) >= 3:
                lo, hi = self._select_repair_window(pp, self.rng)
                child = self._window_repair(child, lo, min(hi, lo + 2))
        return child

    def _crossover(self, p1: Solution, p2: Solution) -> Solution:
        if self.rng.random() >= self.block_crossover_rate:
            return super()._crossover(p1, p2)
        return self._block_crossover(p1, p2)

    # ----------------------------------------------------------- elite polish
    def _postprocess_elite(self, pop: List[Solution], budget: float) -> List[Solution]:
        if not pop or budget <= 0.3:
            return pop
        t_end = min(_time.time() + budget, self._deadline - self._FINAL_RESERVE_S - 2.0)
        elite_n = max(2, min(len(pop) // 4, 8))
        out = list(pop)
        for i in range(elite_n):
            if _time.time() >= t_end:
                break
            sol = out[i]
            progs = sol.scheduled_programs
            if len(progs) < 3:
                continue
            lo, hi = self._select_repair_window(progs, self.rng)
            hi = min(hi, lo + self._adaptive_window_params()["max_segs"] - 1)
            new_sol = self._window_repair(sol, lo, hi)
            if new_sol.total_score > sol.total_score:
                out[i] = new_sol
        out.sort(key=lambda s: s.total_score, reverse=True)
        return out

    def _ga(self, pop: List[Solution], budget: float) -> Solution:
        """Clamp GA wall time to the upgraded scheduler's hard deadline."""
        room = max(0.1, self._deadline - _time.time() - 0.35)
        return super()._ga(pop, min(budget, room))

    def _final_polish(self, best: Solution) -> Solution:
        progs = best.scheduled_programs
        if len(progs) < 4:
            return best
        if self.polish_rounds is not None:
            rounds = self.polish_rounds
        else:
            rounds = 2 if self.n_channels > 1500 else 3
        cur = best
        for _ in range(rounds):
            if not self._deadline_ok(margin=self._FINAL_RESERVE_S):
                break
            lo, hi = self._select_repair_window(progs, self.rng)
            params = self._adaptive_window_params()
            hi = min(hi, lo + max(1, params["max_segs"] // 2))
            nxt = self._window_repair(cur, lo, hi)
            if nxt.total_score > cur.total_score:
                cur = nxt
                progs = cur.scheduled_programs
        return cur

    # ============================================================== entry ====
    def generate_solution(self) -> Solution:
        self._t0 = _time.time()
        self._deadline = self._t0 + self.time_limit - self._FINAL_RESERVE_S

        if self.verbose:
            print(f"\n{'='*60}")
            print(
                f"APEX UPGRADED  (ch={self.n_channels}  "
                f"budget={self.time_limit:.0f}s  pop={self.pop_size})"
            )
            if self._total_pref:
                print(
                    f"Prefs: {self._total_pref}  genres={self._pref_genres}  "
                    f"pen={self._pen} ratio={self._pen_ratio:.2f}"
                )
            print("=" * 60)

        constr_budget = max(5.0, self.time_limit * (1.0 - self.ga_frac) - 10.0)
        population = self._seed(constr_budget)

        if not population:
            return Solution([], 0)

        elite_budget = min(
            22.0,
            max(2.5, (self._deadline - _time.time()) * 0.14),
        )
        if elite_budget > 1.0 and self._deadline_ok(margin=6.0):
            population = self._postprocess_elite(population, elite_budget)

        ga_budget = max(0.0, self._deadline - _time.time() - 1.5)
        if ga_budget > 8.0 and len(population) >= 4:
            best = self._ga(population, ga_budget)
        else:
            best = population[0]

        if self._deadline_ok(margin=self._FINAL_RESERVE_S):
            best = self._final_polish(best)

        if self.verbose:
            print(f"\n{'='*60}")
            print(
                f"RESULT  score={best.total_score}  "
                f"segs={len(best.scheduled_programs)}  "
                f"time={self._elapsed():.1f}s"
            )
            print("=" * 60 + "\n")

        return best
