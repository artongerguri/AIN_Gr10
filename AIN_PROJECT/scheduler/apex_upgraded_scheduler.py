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
from utils.debug_breakpoints import debug_breakpoint


class ApexUpgradedScheduler(ApexScheduler):
    """
    Same bounded wall-clock budget as Apex, with:
    - Block / chain identification for crossover and mutations
    - Bounded beam re-optimization inside removed time windows
    - Region-first mutations (weak / switch-heavy / pref-adjacent / random)
    - Explicit post-GA Local Search pass when time remains
    """

    _FINAL_RESERVE_S = 4.0
    _MAX_TIME_LIMIT_S = 600.0

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
        local_search_budget_seconds: Optional[float] = None,
        local_search_fraction: float = 0.10,
        guided_local_search: bool = False,
        guided_penalty_weight: float = 0.35,
        seed: Optional[int] = None,
        lookahead_limit: int = 6,
        density_percentile: int = 25,
        verbose: bool = True,
    ):
        debug_breakpoint(
            "ApexUpgradedScheduler.__init__",
            channels=len(instance_data.channels),
            time_limit_seconds=time_limit_seconds,
            population_size=population_size,
            local_search_budget_seconds=local_search_budget_seconds,
        )
        tl = min(max(1.0, float(time_limit_seconds)), self._MAX_TIME_LIMIT_S)
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
        self._hard_deadline: float = 0.0
        self.destroy_rebuild_rate = destroy_rebuild_rate
        self.block_crossover_rate = block_crossover_rate
        self.polish_rounds = polish_rounds
        self.local_search_budget_seconds = local_search_budget_seconds
        self.local_search_fraction = max(0.0, float(local_search_fraction))
        self.guided_local_search = bool(guided_local_search)
        self.guided_penalty_weight = max(0.0, float(guided_penalty_weight))
        self._gls_penalties: Dict[Tuple[str, str], int] = {}
        self._gls_updates = 0
        self.ga_solution: Optional[Solution] = None
        self.ga_score: int = 0
        self.ga_segments: int = 0
        self.ga_elapsed_s: float = 0.0
        self.ga_ls_solution: Optional[Solution] = None
        self.ga_ls_score: int = 0
        self.ga_ls_segments: int = 0
        self.local_search_elapsed_s: float = 0.0
        self.local_search_attempts: int = 0
        self.local_search_improvements: int = 0
        self.local_search_improvement: int = 0
        (
            self._pref_channel_hits,
            self._pref_channel_max_hits,
        ) = self._build_preference_channel_hits()

    def _deadline_ok(self, margin: float = 0.0) -> bool:
        return _time.time() + margin < self._deadline

    def _phase_time_left(self, reserve: float = 0.0) -> float:
        return max(0.0, self._deadline - _time.time() - reserve)

    def _planned_local_search_budget(self) -> float:
        if self.local_search_budget_seconds is not None:
            return max(0.0, float(self.local_search_budget_seconds))
        if self.time_limit < 30.0:
            return max(0.3, self.time_limit * min(self.local_search_fraction, 0.20))
        return min(45.0, max(8.0, self.time_limit * self.local_search_fraction))

    def _gls_active(self) -> bool:
        return self.guided_local_search and self.guided_penalty_weight > 0.0

    def get_run_metrics(self) -> Dict[str, float]:
        return {
            "ga_score": self.ga_score,
            "ga_segments": self.ga_segments,
            "ga_ls_score": self.ga_ls_score,
            "ga_ls_segments": self.ga_ls_segments,
            "improvement": self.local_search_improvement,
            "improvement_pct": (
                round((self.local_search_improvement / self.ga_score) * 100.0, 4)
                if self.ga_score
                else 0.0
            ),
            "ga_time_s": round(self.ga_elapsed_s, 2),
            "ls_time_s": round(self.local_search_elapsed_s, 2),
            "ls_attempts": self.local_search_attempts,
            "ls_improvements": self.local_search_improvements,
            "gls_updates": self._gls_updates,
        }

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
        debug_breakpoint(
            "ApexUpgradedScheduler._identify_blocks",
            segments=len(sched),
        )
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
        debug_breakpoint(
            "ApexUpgradedScheduler._select_repair_window",
            segments=len(progs),
        )
        n = len(progs)
        if n <= 1:
            return 0, 0
        p = self._adaptive_window_params()
        w = min(p["max_segs"], max(2, n // 4 + 1))
        w = min(w, n)

        if self._gls_active() and self._gls_penalties and rng.random() < 0.45:
            guided_window = self._gls_pick_window(progs, rng)
            if guided_window is not None:
                return guided_window

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

    def _gls_segment_features(
        self, progs: List[Schedule], idx: int
    ) -> List[Tuple[Tuple[str, str], float]]:
        seg = progs[idx]
        features: List[Tuple[Tuple[str, str], float]] = []
        duration = max(self.min_d, seg.end - seg.start)
        typical = max(1.0, self.avg_score_per_min * duration)
        weak_cost = max(0.0, typical - seg.fitness)
        if weak_cost > 0.0:
            features.append((("weak", seg.unique_program_id), weak_cost))

        if idx > 0 and progs[idx - 1].channel_id != seg.channel_id:
            key = f"{progs[idx - 1].channel_id}->{seg.channel_id}"
            features.append((("switch", key), max(1.0, float(self._pen))))

        if self._pw:
            pi = self.prog_by_id.get(seg.unique_program_id)
            if pi:
                prog = pi[0]
                for pref in self._pw:
                    if prog.genre != pref[2]:
                        continue
                    overlap = self._pref_overlap(seg.start, seg.end, pref)
                    near_pref = seg.end > pref[0] - self.min_d and seg.start < pref[1] + self.min_d
                    if 0 < overlap < self.min_d:
                        cost = pref[3] * 0.75
                    elif overlap <= 0 and near_pref:
                        cost = pref[3] * 0.35
                    else:
                        continue
                    key = f"{seg.unique_program_id}:{pref[0]}:{pref[1]}"
                    features.append((("pref", key), max(1.0, float(cost))))

        if not features:
            features.append((("selected", seg.unique_program_id), 1.0))
        return features

    def _gls_segment_pressure(self, progs: List[Schedule], idx: int) -> float:
        if not self._gls_active() or not self._gls_penalties:
            return 0.0
        pressure = 0.0
        for feature, cost in self._gls_segment_features(progs, idx):
            pressure += self._gls_penalties.get(feature, 0) * max(1.0, cost)
        return pressure * self.guided_penalty_weight

    def _gls_ranked_indices(self, progs: List[Schedule]) -> List[Tuple[float, int]]:
        ranked = [
            (self._gls_segment_pressure(progs, idx), idx)
            for idx in range(len(progs))
        ]
        ranked = [(pressure, idx) for pressure, idx in ranked if pressure > 0.0]
        ranked.sort(reverse=True)
        return ranked

    def _gls_pick_window(
        self, progs: List[Schedule], rng
    ) -> Optional[Tuple[int, int]]:
        ranked = self._gls_ranked_indices(progs)
        if not ranked:
            return None
        params = self._adaptive_window_params()
        max_w = min(params["max_segs"], len(progs))
        pool = ranked[: min(6, len(ranked))]
        total = sum(pressure for pressure, _idx in pool)
        if total <= 0.0:
            idx = pool[0][1]
        else:
            pick = rng.random() * total
            acc = 0.0
            idx = pool[-1][1]
            for pressure, cand_idx in pool:
                acc += pressure
                if pick <= acc:
                    idx = cand_idx
                    break
        radius = rng.randint(0, max(1, max_w // 2))
        lo = max(0, idx - radius)
        hi = min(len(progs) - 1, idx + radius)
        return lo, hi

    def _gls_update_penalties(self, sol: Solution) -> None:
        if not self._gls_active():
            return
        progs = sol.scheduled_programs
        if not progs:
            return

        utilities: Dict[Tuple[str, str], float] = {}
        for idx in range(len(progs)):
            for feature, cost in self._gls_segment_features(progs, idx):
                utility = cost / (1 + self._gls_penalties.get(feature, 0))
                utilities[feature] = max(utilities.get(feature, 0.0), utility)

        if not utilities:
            return
        ranked = sorted(utilities.items(), key=lambda item: item[1], reverse=True)
        limit = min(3, len(ranked))
        threshold = ranked[0][1] * 0.92
        for feature, utility in ranked[:limit]:
            if utility < threshold:
                break
            self._gls_penalties[feature] = self._gls_penalties.get(feature, 0) + 1
        self._gls_updates += 1

    def _build_preference_channel_hits(
        self,
    ) -> Tuple[Dict[Tuple[int, int, str], Dict[int, int]], Dict[Tuple[int, int, str], int]]:
        hits: Dict[Tuple[int, int, str], Dict[int, int]] = {}
        max_hits: Dict[Tuple[int, int, str], int] = {}
        for ps, pe, genre, _bonus in self._pw:
            key = (ps, pe, genre)
            per_channel: Dict[int, int] = {}
            for ch_idx, progs in enumerate(self.ch_progs):
                ch_id = self.instance_data.channels[ch_idx].channel_id
                count = 0
                for prog in progs:
                    if prog.genre != genre:
                        continue
                    ss = max(prog.start, ps, self._open)
                    se = min(prog.end, pe, self._close)
                    if se - ss < self.min_d:
                        continue
                    if not self._channel_allowed(ch_idx, ss, se):
                        continue
                    count += 1
                if count:
                    per_channel[ch_id] = count
            if per_channel:
                hits[key] = per_channel
                max_hits[key] = max(per_channel.values())
        return hits, max_hits

    def _pref_key(self, pref: Tuple[int, int, str, int]) -> Tuple[int, int, str]:
        return pref[0], pref[1], pref[2]

    def _pref_overlap(
        self, start: int, end: int, pref: Tuple[int, int, str, int]
    ) -> int:
        return min(end, pref[1]) - max(start, pref[0])

    def _segment_captures_preference(
        self, seg: Schedule, pref: Tuple[int, int, str, int]
    ) -> bool:
        pi = self.prog_by_id.get(seg.unique_program_id)
        if not pi or pi[0].genre != pref[2]:
            return False
        return self._pref_overlap(seg.start, seg.end, pref) >= self.min_d

    def _bonus_channel_weight(
        self, ch_id: int, pref: Tuple[int, int, str, int]
    ) -> float:
        key = self._pref_key(pref)
        hits = self._pref_channel_hits.get(key)
        max_hits = self._pref_channel_max_hits.get(key, 0)
        if not hits or max_hits <= 0:
            return 0.0
        return pref[3] * 0.15 * (hits.get(ch_id, 0) / max_hits)

    def _bonus_candidate_hint(
        self,
        cand,
        target_pref: Tuple[int, int, str, int],
    ) -> float:
        _seg_score, _ch_idx, ch_id, prog, ss, se = cand
        hint = 0.0
        if prog.genre == target_pref[2]:
            ov = self._pref_overlap(ss, se, target_pref)
            if ov >= self.min_d:
                hint += target_pref[3] * 0.85
            elif ov > 0:
                hint += target_pref[3] * 0.25
            hint += self._bonus_channel_weight(ch_id, target_pref)
        return hint

    def _beam_state_priority(
        self,
        state,
        t1: int,
        target_pref: Optional[Tuple[int, int, str, int]],
    ) -> float:
        score, time, prev_ch, _prev_genre, _streak, _segs, _used = state
        rank = score + (min(t1, self._close) - time) * self.avg_score_per_min * 0.018
        if target_pref is not None:
            if time <= target_pref[1] - self.min_d and time < t1:
                rank += target_pref[3] * 0.08
            if prev_ch is not None:
                rank += self._bonus_channel_weight(prev_ch, target_pref) * 0.25
        return rank

    def _preference_bonus_windows(
        self, progs: List[Schedule]
    ) -> List[Tuple[int, int, Tuple[int, int, str, int]]]:
        if not self._pw or not progs:
            return []
        n = len(progs)
        max_w = min(self._adaptive_window_params()["max_segs"], n)
        windows: List[Tuple[int, int, int, int, Tuple[int, int, str, int]]] = []
        seen: Set[Tuple[int, int, Tuple[int, int, str]]] = set()

        for pref in sorted(self._pw, key=lambda p: p[3], reverse=True):
            key = self._pref_key(pref)
            if key not in self._pref_channel_hits:
                continue

            touched: List[int] = []
            missed: List[Tuple[int, int]] = []
            for i, seg in enumerate(progs):
                ov = self._pref_overlap(seg.start, seg.end, pref)
                if ov <= 0:
                    continue
                touched.append(i)
                if not self._segment_captures_preference(seg, pref):
                    missed.append((i, ov))

            if not touched:
                center = 0
                while center < n and progs[center].end <= pref[0]:
                    center += 1
                center = min(center, n - 1)
                target_idxs = [center]
                lo = max(0, center - 1)
                hi = min(n - 1, center + 1)
            else:
                target_idxs = [i for i, ov in missed if ov >= self.min_d]
                if not target_idxs:
                    target_idxs = [i for i, _ov in missed]
                if not target_idxs:
                    continue
                lo = max(0, min(target_idxs) - 1)
                hi = min(n - 1, max(target_idxs) + 1)

            if hi - lo + 1 > max_w:
                mid_t = (pref[0] + pref[1]) // 2
                center = min(
                    target_idxs,
                    key=lambda i: abs(((progs[i].start + progs[i].end) // 2) - mid_t),
                )
                half = max_w // 2
                lo = max(0, center - half)
                hi = min(n - 1, lo + max_w - 1)
                lo = max(0, hi - max_w + 1)

            dedupe_key = (lo, hi, key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            windows.append((pref[3], len(target_idxs), lo, hi, pref))

        windows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(lo, hi, pref) for _bonus, _targets, lo, hi, pref in windows]

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
        target_pref: Optional[Tuple[int, int, str, int]] = None,
    ) -> Tuple[List[Schedule], int]:
        """
        Layered bounded beam from t0; each scheduled segment must end <= t1.
        Returns (new_segments, sum of segment scores in the gap).
        """
        debug_breakpoint(
            "ApexUpgradedScheduler._beam_fill_gap",
            gap=(t0, t1),
            prev_channel=prev_ch,
            prev_genre=prev_genre,
            frozen_used=len(frozen_used),
            beam_width=params.get("beam_w"),
            target_pref=target_pref,
        )
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

                # LS repair may improve by leaving part of a gap idle.
                wait_state = None
                idx = bisect.bisect_right(self.times, time)
                if idx < len(self.times):
                    wait_to = min(self.times[idx], t1, self._close)
                    if wait_to > time:
                        wait_state = (
                            score,
                            wait_to,
                            pch,
                            pgr,
                            gst,
                            segs[:],
                            set(used),
                        )

                cands = self._gen(time, pch, pgr, gst, used)
                if wait_state is not None:
                    nxt.append(wait_state)

                if not cands:
                    if wait_state is None and score > best_score:
                        best_score = score
                        best_segs = segs[:]
                    continue

                if target_pref is None:
                    cands.sort(key=lambda x: x[0], reverse=True)
                else:
                    ranked = self._rank(cands, time, pch, pgr, gst, "max_pref", used)
                    ranked.sort(
                        key=lambda x: x[0]
                        + self._bonus_candidate_hint(x[1], target_pref),
                        reverse=True,
                    )
                    cands = [c for _rank_score, c in ranked]
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
                    key=lambda st: self._beam_state_priority(st, t1, target_pref),
                    reverse=True,
                )
                for st in br[:beam_w]:
                    nxt.append(st)

            if expansions >= max_exp or _time.time() >= outer_deadline:
                break

            if not nxt:
                break
            nxt.sort(
                key=lambda st: self._beam_state_priority(st, t1, target_pref),
                reverse=True,
            )
            cap = max(beam_w * 6, beam_w)
            layer = nxt[:cap]

        for score, _, _, _, _, segs, _ in layer:
            if score > best_score:
                best_score = score
                best_segs = segs
        return best_segs, best_score

    def _window_repair(
        self,
        sol: Solution,
        lo: int,
        hi: int,
        target_pref: Optional[Tuple[int, int, str, int]] = None,
    ) -> Solution:
        """Remove segments [lo..hi] and re-optimize the gap with bounded beam."""
        debug_breakpoint(
            "ApexUpgradedScheduler._window_repair",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
            window=(lo, hi),
        )
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
            t0, t1, prev_ch, prev_genre, gs, used, params, outer_deadline, target_pref
        )
        if not mid and not suffix:
            tail = self._construct(
                strategy="max_pref" if target_pref is not None else "density",
                alpha=0.05,
                prefix=prefix if prefix else None,
            )
            return tail

        raw = prefix + mid + suffix
        return self._repair(raw)

    # -------------------------------------------------------------- mutations
    def _destroy_rebuild_mutation(self, sol: Solution) -> Solution:
        debug_breakpoint(
            "ApexUpgradedScheduler._destroy_rebuild_mutation",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
        )
        progs = sol.scheduled_programs
        if len(progs) < 2:
            strat = self.rng.choice(["density", "chain", "balanced"])
            return self._construct(strategy=strat, alpha=0.15, extend=self.rng.random() < 0.5)

        lo, hi = self._select_repair_window(progs, self.rng)
        return self._window_repair(sol, lo, hi)

    def _mutate(self, sol: Solution) -> Solution:
        debug_breakpoint(
            "ApexUpgradedScheduler._mutate",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
            destroy_rebuild_rate=self.destroy_rebuild_rate,
        )
        if len(sol.scheduled_programs) < 2:
            return super()._mutate(sol)
        if self.rng.random() < self.destroy_rebuild_rate:
            return self._destroy_rebuild_mutation(sol)
        return super()._mutate(sol)

    # --------------------------------------------------------------- crossover
    def _block_crossover(self, p1: Solution, p2: Solution) -> Solution:
        debug_breakpoint(
            "ApexUpgradedScheduler._block_crossover",
            parent1_score=p1.total_score,
            parent2_score=p2.total_score,
            parent1_segments=len(p1.scheduled_programs),
            parent2_segments=len(p2.scheduled_programs),
        )
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
        debug_breakpoint(
            "ApexUpgradedScheduler._crossover",
            parent1_score=p1.total_score,
            parent2_score=p2.total_score,
            block_crossover_rate=self.block_crossover_rate,
        )
        if self.rng.random() >= self.block_crossover_rate:
            return super()._crossover(p1, p2)
        return self._block_crossover(p1, p2)

    # ----------------------------------------------------------- elite polish
    def _postprocess_elite(self, pop: List[Solution], budget: float) -> List[Solution]:
        debug_breakpoint(
            "ApexUpgradedScheduler._postprocess_elite",
            population=len(pop),
            budget=round(budget, 3),
            best=pop[0].total_score if pop else None,
        )
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
        debug_breakpoint(
            "ApexUpgradedScheduler._ga",
            population=len(pop),
            requested_budget=round(budget, 3),
        )
        room = max(0.1, self._deadline - _time.time() - 0.35)
        return super()._ga(pop, min(budget, room))

    def _add_window(
        self,
        windows: List[Tuple[int, int]],
        seen: Set[Tuple[int, int]],
        lo: int,
        hi: int,
        n: int,
    ) -> None:
        if n <= 0:
            return
        lo = max(0, min(lo, n - 1))
        hi = max(lo, min(hi, n - 1))
        key = (lo, hi)
        if key not in seen:
            seen.add(key)
            windows.append(key)

    def _local_search_windows(self, progs: List[Schedule]) -> List[Tuple[int, int]]:
        debug_breakpoint(
            "ApexUpgradedScheduler._local_search_windows",
            segments=len(progs),
        )
        n = len(progs)
        windows: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        if n < 2:
            return windows

        params = self._adaptive_window_params()
        max_w = min(params["max_segs"], n)

        for w in range(1, max_w + 1):
            lo = self._weak_region(progs, w)
            self._add_window(windows, seen, lo, lo + w - 1, n)

        worst = sorted(range(n), key=lambda i: progs[i].fitness)[: min(8, n)]
        for idx in worst:
            self._add_window(windows, seen, idx, idx, n)
            self._add_window(windows, seen, idx - 1, idx + 1, n)
            if max_w >= 4:
                self._add_window(windows, seen, idx - 2, idx + 2, n)

        for w in range(2, max_w + 1):
            lo = self._switch_heavy_region(progs, w)
            self._add_window(windows, seen, lo, lo + w - 1, n)

        if self._pw:
            for w in range(2, max_w + 1):
                lo = self._pref_adjacent_region(progs, w, self.rng)
                self._add_window(windows, seen, lo, lo + w - 1, n)

        if self._gls_active() and self._gls_penalties:
            for _pressure, idx in self._gls_ranked_indices(progs)[: min(8, n)]:
                self._add_window(windows, seen, idx, idx, n)
                self._add_window(windows, seen, idx - 1, idx + 1, n)
                if max_w >= 4:
                    self._add_window(windows, seen, idx - 2, idx + 2, n)

        natural_times = set(self._natural_cut_times())
        for i, seg in enumerate(progs):
            if seg.start in natural_times or seg.end in natural_times:
                self._add_window(windows, seen, i - 1, i + 1, n)

        return windows

    def _local_search_cut_points(self, progs: List[Schedule]) -> List[int]:
        """Tail rebuild cut points: keep progs[:cut] and reconstruct the rest."""
        debug_breakpoint(
            "ApexUpgradedScheduler._local_search_cut_points",
            segments=len(progs),
        )
        n = len(progs)
        if n < 2:
            return []

        cuts: List[int] = []
        seen: Set[int] = set()

        def add(cut: int) -> None:
            cut = max(1, min(cut, n - 1))
            if cut not in seen:
                seen.add(cut)
                cuts.append(cut)

        params = self._adaptive_window_params()
        max_w = min(params["max_segs"], n)

        for w in range(1, max_w + 1):
            add(self._weak_region(progs, w))

        worst = sorted(range(n), key=lambda i: progs[i].fitness)[: min(10, n)]
        for idx in worst:
            add(idx)
            add(idx - 1)

        for w in range(2, max_w + 1):
            add(self._switch_heavy_region(progs, w))

        if self._pw:
            for w in range(2, max_w + 1):
                add(self._pref_adjacent_region(progs, w, self.rng))

        if self._gls_active() and self._gls_penalties:
            for _pressure, idx in self._gls_ranked_indices(progs)[: min(10, n)]:
                add(idx)
                add(idx - 1)

        natural_times = set(self._natural_cut_times())
        for i, seg in enumerate(progs):
            if seg.start in natural_times or seg.end in natural_times:
                add(i)

        stride = max(2, n // 8)
        for cut in range(stride, n, stride):
            add(cut)

        return cuts

    def _local_search_profiles(self) -> List[Tuple[str, float, Optional[int], bool]]:
        profiles: List[Tuple[str, float, Optional[int], bool]] = [
            ("density", 0.0, None, False),
            ("balanced", 0.0, None, False),
            ("continuation", 0.0, None, False),
            ("chain", 0.0, None, True),
            ("density", 0.05, None, False),
            ("balanced", 0.08, None, False),
            ("chain", 0.05, None, True),
        ]
        if self._pref_genres:
            profiles.extend(
                [
                    ("max_pref", 0.0, None, False),
                    ("max_pref", 0.08, None, False),
                    ("density", 0.0, 1, False),
                    ("balanced", 0.0, 1, False),
                    ("chain", 0.0, 1, True),
                ]
            )
        if self._large:
            profiles.extend(
                [
                    ("v_density", 0.0, None, False),
                    ("v_balanced", 0.0, None, False),
                    ("v_continuation", 0.0, None, False),
                ]
            )
        return profiles

    def _tail_rebuild(self, sol: Solution) -> Solution:
        """Relaxed Local Search: rebuild the suffix after promising cut points."""
        debug_breakpoint(
            "ApexUpgradedScheduler._tail_rebuild",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
        )
        progs = sol.scheduled_programs
        if len(progs) < 3 or not self._deadline_ok(margin=0.10):
            return sol

        best = sol
        profiles = self._local_search_profiles()
        cuts = self._local_search_cut_points(progs)

        for cut in cuts:
            if not self._deadline_ok(margin=0.10):
                break
            prefix = progs[:cut]
            for strategy, alpha, mc_override, extend in profiles:
                if not self._deadline_ok(margin=0.10):
                    break
                budget = min(0.65, max(0.08, self._phase_time_left(reserve=0.08)))
                self.local_search_attempts += 1
                cand = self._construct(
                    strategy=strategy,
                    alpha=alpha,
                    mc_override=mc_override,
                    prefix=prefix,
                    time_limit=budget,
                    extend=extend,
                )
                if cand.total_score > best.total_score:
                    best = cand
                    prefix = cand.scheduled_programs[: min(cut, len(cand.scheduled_programs))]

        return best

    def _preference_anchored_window_repair(self, sol: Solution) -> Solution:
        """Repair windows that overlap preference bonuses but are not capturing them."""
        debug_breakpoint(
            "ApexUpgradedScheduler._preference_anchored_window_repair",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
        )
        if not self._pw or len(sol.scheduled_programs) < 2:
            return sol

        for lo, hi, pref in self._preference_bonus_windows(sol.scheduled_programs):
            if not self._deadline_ok(margin=0.08):
                break
            self.local_search_attempts += 1
            cand = self._window_repair(sol, lo, hi, target_pref=pref)
            if cand.total_score > sol.total_score:
                return cand
        return sol

    def _preference_greedy_search(self, sol: Solution) -> Solution:
        """
        OPTIMIZIMI #3: Preference-greedy local search.
        
        Kërkohje lokal i fokusuar në segmentet e preferencës kohore:
        1. Identifikon zonat rreth preference bonus windows
        2. Për çdo segment me genre të përshtatshëm, provojnë ta zhvendosësh 
           në pozicione më të mira brenda preference window
        3. Përdor window repair për të rimotorizuar segmentet rreth preferencave
        
        Qëllimi: Siguroj që të marrim sa më shumë preference bonuses të mundshëm.
        """
        debug_breakpoint(
            "ApexUpgradedScheduler._preference_greedy_search",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
        )
        
        if not self._pw or len(sol.scheduled_programs) < 2:
            return sol

        best = sol
        
        # Për çdo preference window, provojnë ta optimizojmë atë zonë
        for pref_idx, pref in enumerate(self._pw):
            if not self._deadline_ok(margin=0.10):
                break
            
            pref_start, pref_end, pref_genre, pref_bonus = pref
            
            # Gjej segmentet në orar aktual që kanë genre të përputhet
            progs = best.scheduled_programs
            n = len(progs)
            
            matching_indices = []
            for i, seg in enumerate(progs):
                pi = self.prog_by_id.get(seg.unique_program_id)
                if pi and pi[0].genre == pref_genre:
                    matching_indices.append(i)
            
            if not matching_indices:
                continue
            
            # Gjej segmentet që tashmë përputhen me preferencën
            capturing = []
            not_capturing = []
            for idx in matching_indices:
                seg = progs[idx]
                ov = self._pref_overlap(seg.start, seg.end, pref)
                if ov >= self.min_d:
                    capturing.append((idx, ov))
                else:
                    not_capturing.append(idx)
            
            # Nëse tashmë i kemi mirë, shko më tej
            if len(capturing) >= 2:
                continue
            
            # Nëse janë shumë afër, kryej window repair rreth preferencave
            if not_capturing or capturing:
                # Gjej dritaren rreth preferencës
                touch_indices = [idx for idx, _ in capturing] + not_capturing
                if not touch_indices:
                    continue
                
                lo = min(touch_indices)
                hi = max(touch_indices)
                
                # Zgjeroje pak dritaren për të lejuar rehashje më të mirë
                lo = max(0, lo - 1)
                hi = min(n - 1, hi + 1)
                
                if lo <= hi:
                    self.local_search_attempts += 1
                    cand = self._window_repair(best, lo, hi, target_pref=pref)
                    if cand.total_score > best.total_score:
                        best = cand
                        # Rilexo pas ndryshimit
                        continue
        
        return best

    def _alns_initial_weights(self) -> Dict[str, float]:
        weights = {
            "boundary": 0.8,
            "weak_window": 1.1,
            "worst_window": 1.1,
            "switch_window": 0.9,
            "random_window": 0.8,
            "tail": 0.9,
            "stochastic": 0.6,
        }
        if self._pw:
            weights["preference"] = 1.7
        if self._gls_active() and self._gls_penalties:
            weights["guided_window"] = 1.4
        return weights

    def _alns_pick_operator(self, weights: Dict[str, float]) -> str:
        total = sum(max(0.0, w) for w in weights.values())
        if total <= 0.0:
            return self.rng.choice(list(weights))
        pick = self.rng.random() * total
        acc = 0.0
        for name, weight in weights.items():
            acc += max(0.0, weight)
            if pick <= acc:
                return name
        return next(reversed(weights))

    def _alns_update_weight(
        self, weights: Dict[str, float], name: str, gain: int
    ) -> None:
        current = weights.get(name, 1.0)
        if gain > 0:
            typical_gain = max(1.0, self.avg_score_per_min * self.min_d)
            reward = 0.75 + min(2.5, gain / typical_gain)
            weights[name] = min(8.0, current * 1.08 + reward)
        else:
            weights[name] = max(0.25, current * 0.94)

    def _alns_window_repair(self, sol: Solution, mode: str) -> Solution:
        progs = sol.scheduled_programs
        n = len(progs)
        if n < 2:
            return sol

        params = self._adaptive_window_params()
        max_w = min(params["max_segs"], n)
        if mode == "weak_window":
            w = self.rng.randint(1, max_w)
            lo = self._weak_region(progs, w)
            hi = min(n - 1, lo + w - 1)
        elif mode == "worst_window":
            worst = sorted(range(n), key=lambda i: progs[i].fitness)[: min(6, n)]
            idx = self.rng.choice(worst)
            radius = self.rng.randint(0, max(1, max_w // 2))
            lo = max(0, idx - radius)
            hi = min(n - 1, idx + radius)
        elif mode == "switch_window":
            if n < 3:
                lo, hi = self._select_repair_window(progs, self.rng)
            else:
                w = self.rng.randint(2, max(2, max_w))
                lo = self._switch_heavy_region(progs, w)
                hi = min(n - 1, lo + w - 1)
        else:
            lo, hi = self._select_repair_window(progs, self.rng)

        self.local_search_attempts += 1
        return self._window_repair(sol, lo, hi)

    def _guided_window_repair(self, sol: Solution) -> Solution:
        progs = sol.scheduled_programs
        if len(progs) < 2:
            return sol
        picked = self._gls_pick_window(progs, self.rng)
        if picked is None:
            return sol
        lo, hi = picked
        self.local_search_attempts += 1
        return self._window_repair(sol, lo, hi)

    def _alns_apply_operator(self, sol: Solution, name: str) -> Solution:
        if name == "preference":
            return self._preference_anchored_window_repair(sol)
        if name == "guided_window":
            return self._guided_window_repair(sol)
        if name in {"weak_window", "worst_window", "switch_window", "random_window"}:
            return self._alns_window_repair(sol, name)
        if name == "tail":
            return self._tail_rebuild(sol)
        if name == "stochastic":
            return self._stochastic_tail_rebuild(sol)
        self.local_search_attempts += 1
        return self._boundary_expand(sol)

    def _deterministic_first_improvement(
        self, sol: Solution
    ) -> Tuple[Solution, Optional[str]]:
        expanded = self._boundary_expand(sol)
        self.local_search_attempts += 1
        if expanded.total_score > sol.total_score:
            return expanded, "boundary"

        pref_repaired = self._preference_anchored_window_repair(sol)
        if pref_repaired.total_score > sol.total_score:
            return pref_repaired, "preference"

        for lo, hi in self._local_search_windows(sol.scheduled_programs):
            if not self._deadline_ok(margin=0.08):
                break
            self.local_search_attempts += 1
            cand = self._window_repair(sol, lo, hi)
            if cand.total_score > sol.total_score:
                return cand, "weak_window"

        tail = self._tail_rebuild(sol)
        if tail.total_score > sol.total_score:
            return tail, "tail"

        stochastic = self._stochastic_tail_rebuild(sol)
        if stochastic.total_score > sol.total_score:
            return stochastic, "stochastic"

        return sol, None

    def _boundary_expand(self, sol: Solution) -> Solution:
        """Expand segment boundaries into adjacent idle gaps when score improves."""
        debug_breakpoint(
            "ApexUpgradedScheduler._boundary_expand",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
        )
        progs = sol.scheduled_programs
        if not progs:
            return sol

        out: List[Schedule] = []
        total = 0
        changed = False
        prev_ch: Optional[int] = None

        for i, seg in enumerate(progs):
            pi = self.prog_by_id.get(seg.unique_program_id)
            if not pi:
                out.append(seg)
                total += seg.fitness
                prev_ch = seg.channel_id
                continue

            prog, ch_idx = pi
            prev_end = out[-1].end if out else self._open
            next_start = progs[i + 1].start if i + 1 < len(progs) else self._close

            best_start = seg.start
            best_end = seg.end
            best_score = self._calc_score(prog, ch_idx, seg.start, seg.end, prev_ch)

            cand_start = max(prev_end, prog.start, self._open)
            cand_end = min(next_start, prog.end, self._close)

            candidates = [
                (cand_start, seg.end),
                (seg.start, cand_end),
                (cand_start, cand_end),
            ]
            for ns, ne in candidates:
                if ns > seg.start or ne < seg.end:
                    continue
                if ne - ns < self.min_d:
                    continue
                if not self._channel_allowed(ch_idx, ns, ne):
                    continue
                sc = self._calc_score(prog, ch_idx, ns, ne, prev_ch)
                if sc > best_score:
                    best_start, best_end, best_score = ns, ne, sc

            if best_start != seg.start or best_end != seg.end or best_score != seg.fitness:
                changed = True
                out.append(
                    Schedule(
                        seg.program_id,
                        seg.channel_id,
                        best_start,
                        best_end,
                        best_score,
                        seg.unique_program_id,
                    )
                )
            else:
                out.append(seg)

            total += out[-1].fitness
            prev_ch = out[-1].channel_id

        return Solution(out, total) if changed and total > sol.total_score else sol

    def _stochastic_tail_rebuild(self, sol: Solution) -> Solution:
        """Randomized ILS-style suffix rebuilds for escaping shallow local optima."""
        debug_breakpoint(
            "ApexUpgradedScheduler._stochastic_tail_rebuild",
            score=sol.total_score,
            segments=len(sol.scheduled_programs),
        )
        progs = sol.scheduled_programs
        n = len(progs)
        if n < 4 or not self._deadline_ok(margin=0.15):
            return sol

        best = sol
        base_cuts = self._local_search_cut_points(progs)
        strategies = ["density", "balanced", "continuation", "chain"]
        if self._pref_genres:
            strategies.append("max_pref")
        if self._large:
            strategies.extend(["v_density", "v_balanced", "v_continuation"])

        max_tries = 160 if self.n_channels <= 100 else 70
        tries = 0
        while tries < max_tries and self._deadline_ok(margin=0.15):
            tries += 1
            if base_cuts and self.rng.random() < 0.65:
                cut = self.rng.choice(base_cuts)
            else:
                lo = max(1, n // 8)
                hi = max(lo, n - 2)
                cut = self.rng.randint(lo, hi)

            prefix = best.scheduled_programs[: min(cut, len(best.scheduled_programs) - 1)]
            strategy = self.rng.choice(strategies)
            alpha = self.rng.choice([0.0, 0.03, 0.08, 0.15, 0.25, 0.35, 0.50])
            mc_override = self.rng.choice([None, 1]) if self._pref_genres and self.rng.random() < 0.35 else None
            extend = strategy == "chain" or self.rng.random() < 0.20
            budget = min(0.75, max(0.08, self._phase_time_left(reserve=0.10)))

            self.local_search_attempts += 1
            cand = self._construct(
                strategy=strategy,
                alpha=alpha,
                mc_override=mc_override,
                prefix=prefix,
                time_limit=budget,
                extend=extend,
            )
            if cand.total_score > best.total_score:
                best = cand
                progs = best.scheduled_programs
                n = len(progs)
                base_cuts = self._local_search_cut_points(progs)

        return best

    def _vnd_window_cap(self) -> int:
        if self.n_channels > 5000:
            return 3
        if self.n_channels > 2000:
            return 4
        if self.n_channels > 500:
            return 5
        if self.n_channels > 100:
            return 7
        return 12

    def _vnd_tail_rebuild(self, sol: Solution) -> Solution:
        progs = sol.scheduled_programs
        if len(progs) < 4 or not self._deadline_ok(margin=0.12):
            return sol

        left = self._phase_time_left(reserve=0.10)
        if self.n_channels <= 100:
            cut_cap = min(10, max(4, int(left * 5)))
            profile_cap = 8
        else:
            cut_cap = min(4, max(2, int(left * 2)))
            profile_cap = 4
        cuts = self._local_search_cut_points(progs)[:cut_cap]
        profiles = self._local_search_profiles()[:profile_cap]

        for cut in cuts:
            prefix = progs[:cut]
            for strategy, alpha, mc_override, extend in profiles:
                if not self._deadline_ok(margin=0.12):
                    return sol
                budget = min(
                    0.25 if self.n_channels <= 100 else 0.15,
                    max(0.05, self._phase_time_left(reserve=0.10)),
                )
                self.local_search_attempts += 1
                cand = self._construct(
                    strategy=strategy,
                    alpha=alpha,
                    mc_override=mc_override,
                    prefix=prefix,
                    time_limit=budget,
                    extend=extend,
                )
                if cand.total_score > sol.total_score:
                    return cand
        return sol

    def _vnd_first_improvement(
        self, sol: Solution, allow_tail: bool
    ) -> Tuple[Solution, Optional[str]]:
        self.local_search_attempts += 1
        cand = self._boundary_expand(sol)
        if cand.total_score > sol.total_score:
            return cand, "boundary"

        if self._pw:
            cand = self._preference_anchored_window_repair(sol)
            if cand.total_score > sol.total_score:
                return cand, "preference"

        for lo, hi in self._local_search_windows(sol.scheduled_programs)[: self._vnd_window_cap()]:
            if not self._deadline_ok(margin=0.08):
                break
            self.local_search_attempts += 1
            cand = self._window_repair(sol, lo, hi)
            if cand.total_score > sol.total_score:
                return cand, "window"

        if allow_tail and self._phase_time_left(reserve=0.12) > 0.35:
            cand = self._vnd_tail_rebuild(sol)
            if cand.total_score > sol.total_score:
                return cand, "tail"

        return sol, None

    def _ils_perturb(self, sol: Solution, kick: int) -> Solution:
        progs = sol.scheduled_programs
        n = len(progs)
        if n < 4 or not self._deadline_ok(margin=0.10):
            return sol

        mode = kick % 3
        if mode == 0:
            worst = sorted(range(n), key=lambda i: progs[i].fitness)[: min(8, n)]
            idx = self.rng.choice(worst)
            radius = 1 + min(4, kick // 2)
            lo = max(0, idx - radius)
            hi = min(n - 1, idx + radius)
            self.local_search_attempts += 1
            return self._window_repair(sol, lo, hi)

        if mode == 1:
            params = self._adaptive_window_params()
            w = min(n, max(2, params["max_segs"]))
            if n >= 3 and self.rng.random() < 0.5:
                lo = self._switch_heavy_region(progs, min(w, n))
            else:
                lo = self.rng.randint(0, max(0, n - w))
            hi = min(n - 1, lo + w - 1)
            self.local_search_attempts += 1
            return self._window_repair(sol, lo, hi)

        lo = max(1, n // 5)
        hi = max(lo, n - 2)
        cut = self.rng.randint(lo, hi)
        strategy = self.rng.choice(["density", "balanced", "continuation", "chain"])
        if self._pref_genres and self.rng.random() < 0.35:
            strategy = "max_pref"
        budget = min(
            0.45 if self.n_channels <= 100 else 0.25,
            max(0.05, self._phase_time_left(reserve=0.10)),
        )
        self.local_search_attempts += 1
        return self._construct(
            strategy=strategy,
            alpha=self.rng.choice([0.08, 0.15, 0.25]),
            prefix=progs[:cut],
            time_limit=budget,
            extend=strategy == "chain",
        )

    def _solution_signature(self, sol: Solution) -> Tuple[str, ...]:
        return tuple(s.unique_program_id for s in sol.scheduled_programs)

    def _vnd_ils_from(
        self,
        start_sol: Solution,
        max_vnd_accepts: int,
        max_ils_kicks: int,
    ) -> Solution:
        cur = start_sol
        vnd_accepts = 0
        ils_kicks = 0
        while self._deadline_ok(margin=0.08):
            improved = False

            while vnd_accepts < max_vnd_accepts and self._deadline_ok(margin=0.08):
                before_score = cur.total_score
                cand, _op_name = self._vnd_first_improvement(
                    cur,
                    allow_tail=ils_kicks == 0,
                )
                gain = int(cand.total_score - before_score)
                if gain <= 0:
                    break
                cur = cand
                self.local_search_improvements += 1
                vnd_accepts += 1
                improved = True

            if improved:
                continue

            if ils_kicks >= max_ils_kicks or not self._deadline_ok(margin=0.12):
                break

            self._gls_update_penalties(cur)
            kicked = self._ils_perturb(cur, ils_kicks)
            probe = kicked
            for _ in range(3):
                if not self._deadline_ok(margin=0.08):
                    break
                cand, _op_name = self._vnd_first_improvement(probe, allow_tail=False)
                if cand.total_score <= probe.total_score:
                    break
                probe = cand

            if probe.total_score > cur.total_score:
                cur = probe
                self.local_search_improvements += 1
                vnd_accepts += 1
            ils_kicks += 1

        return cur

    def _exact_small_dp_search(self) -> Solution:
        """Exact-style intensification for small/medium TV instances."""
        if (
            self.n_channels > 30
            or len(self.prog_by_id) > 700
            or len(self.times) > 90
        ):
            return Solution([], 0)
        if not self._deadline_ok(margin=0.20):
            return Solution([], 0)

        uid_end = {uid: prog.end for uid, (prog, _ch_idx) in self.prog_by_id.items()}
        memo: Dict[Tuple[int, Optional[int], str, int, Tuple[str, ...]], int] = {}
        choice: Dict[
            Tuple[int, Optional[int], str, int, Tuple[str, ...]],
            Tuple,
        ] = {}

        def active_at(t: int, used) -> Tuple[str, ...]:
            return tuple(sorted(uid for uid in used if uid_end.get(uid, 0) > t))

        def solve(
            t: int,
            prev_ch: Optional[int],
            prev_genre: str,
            streak: int,
            used_active: Tuple[str, ...],
        ) -> int:
            if not self._deadline_ok(margin=0.05) or t >= self._close:
                return 0

            used_active = active_at(t, used_active)
            key = (t, prev_ch, prev_genre, streak, used_active)
            if key in memo:
                return memo[key]

            used = set(used_active)
            best_score = 0
            best_choice = None

            idx = bisect.bisect_right(self.times, t)
            if idx < len(self.times):
                nt = self.times[idx]
                if t < nt <= self._close:
                    wait_used = active_at(nt, used_active)
                    val = solve(nt, prev_ch, prev_genre, streak, wait_used)
                    if val > best_score:
                        best_score = val
                        best_choice = ("wait", nt)

            for seg_sc, ch_idx, ch_id, prog, ss, se in self._get_candidates(
                t, prev_ch, prev_genre, streak, used
            ):
                if ss < t or se <= ss:
                    continue
                ns = streak + 1 if prog.genre == prev_genre else 1
                if ns > self._mc:
                    continue
                new_used = set(used)
                if prog.end > se:
                    new_used.add(prog.unique_id)
                next_used = active_at(se, new_used)
                val = seg_sc + solve(se, ch_id, prog.genre, ns, next_used)
                if val > best_score:
                    best_score = val
                    best_choice = (
                        "take",
                        seg_sc,
                        ch_id,
                        prog,
                        ss,
                        se,
                        next_used,
                        ns,
                    )

            memo[key] = best_score
            if best_choice is not None:
                choice[key] = best_choice
            return best_score

        total = solve(self._open, None, "", 0, tuple())
        if total <= 0:
            return Solution([], 0)

        out: List[Schedule] = []
        t = self._open
        prev_ch: Optional[int] = None
        prev_genre = ""
        streak = 0
        used_active: Tuple[str, ...] = tuple()

        while t < self._close and self._deadline_ok(margin=0.02):
            used_active = active_at(t, used_active)
            key = (t, prev_ch, prev_genre, streak, used_active)
            action = choice.get(key)
            if action is None:
                break
            if action[0] == "wait":
                t = action[1]
                used_active = active_at(t, used_active)
                continue

            _kind, seg_sc, ch_id, prog, ss, se, next_used, ns = action
            out.append(
                Schedule(
                    prog.program_id,
                    ch_id,
                    ss,
                    se,
                    seg_sc,
                    prog.unique_id,
                )
            )
            t = se
            prev_ch = ch_id
            prev_genre = prog.genre
            streak = ns
            used_active = next_used

        return Solution(out, sum(s.fitness for s in out))

    def _global_beam_intensify(self) -> Solution:
        """Moderate global beam pass for medium instances where full DP is too large."""
        if (
            self.n_channels <= 30
            or self.n_channels > 200
            or len(self.prog_by_id) > 2500
            or len(self.times) > 120
        ):
            return Solution([], 0)
        if self._phase_time_left(reserve=0.20) < 1.5:
            return Solution([], 0)

        old_width = self.beam_width
        old_lookahead = self.lookahead_limit
        try:
            self.beam_width = 220 if self._phase_time_left(reserve=0.20) >= 2.2 else 140
            self.lookahead_limit = max(self.lookahead_limit, 30)
            return self._beam_search_core()
        finally:
            self.beam_width = old_width
            self.lookahead_limit = old_lookahead

    def _deep_intensify(self, sol: Solution) -> Solution:
        """Use remaining LS time for stronger escapes after fast VND/ILS stalls."""
        if not self._deadline_ok(margin=0.20):
            return sol

        best = sol
        if self.n_channels <= 100:
            operators = ["preference_greedy", "tail", "stochastic", "worst_window", "switch_window", "random_window"]
            stale_limit = 8
        else:
            operators = ["preference_greedy", "worst_window", "switch_window", "tail"]
            stale_limit = 3
        if self._pw:
            operators.insert(0, "preference")
        if self._gls_active() and self._gls_penalties:
            operators.insert(0, "guided_window")

        stale = 0
        op_idx = 0
        while stale < stale_limit and self._deadline_ok(margin=0.15):
            name = operators[op_idx % len(operators)]
            op_idx += 1
            before = best.total_score

            if name == "preference_greedy":
                cand = self._preference_greedy_search(best)
            elif name == "tail":
                cand = self._tail_rebuild(best)
            elif name == "stochastic":
                cand = self._stochastic_tail_rebuild(best)
            elif name == "preference":
                cand = self._preference_anchored_window_repair(best)
            elif name == "guided_window":
                cand = self._guided_window_repair(best)
            else:
                cand = self._alns_window_repair(best, name)

            if cand.total_score > before:
                best = cand
                self.local_search_improvements += 1
                stale = 0
            else:
                self._gls_update_penalties(best)
                stale += 1

        return best

    def _local_search(
        self,
        best: Solution,
        budget: float,
        start_pool: Optional[List[Solution]] = None,
    ) -> Solution:
        """Post-GA VND plus short ILS kicks, both bounded by wall-clock time."""
        debug_breakpoint(
            "ApexUpgradedScheduler._local_search.start",
            initial_score=best.total_score,
            segments=len(best.scheduled_programs),
            budget=round(budget, 3),
        )
        if budget <= 0.05 or len(best.scheduled_programs) < 2:
            return best

        start = _time.time()
        self.local_search_attempts = 0
        self.local_search_improvements = 0
        deadline = min(self._hard_deadline - 0.10, start + budget)
        if deadline <= start:
            return best

        self._deadline = deadline
        cur = best

        exact = self._exact_small_dp_search()
        if exact.total_score > cur.total_score:
            cur = exact
            self.local_search_improvements += 1

        beam_sol = self._global_beam_intensify()
        if beam_sol.total_score > cur.total_score:
            cur = beam_sol
            self.local_search_improvements += 1

        if self.polish_rounds is not None:
            max_vnd_accepts = max(3, int(self.polish_rounds))
        else:
            max_vnd_accepts = 12 if self.n_channels <= 100 else 7
        max_ils_kicks = 6 if self.n_channels <= 100 else 3

        starts = [best]
        if start_pool:
            seen = {self._solution_signature(best)}
            pool_cap = 5 if self.n_channels <= 100 else 3
            for sol in sorted(start_pool, key=lambda s: s.total_score, reverse=True):
                if len(starts) >= pool_cap:
                    break
                sig = self._solution_signature(sol)
                if sig in seen:
                    continue
                seen.add(sig)
                starts.append(sol)

        for idx, sol in enumerate(starts):
            if not self._deadline_ok(margin=0.08):
                break
            kicks = max_ils_kicks if idx == 0 else max(2, max_ils_kicks // 2)
            accepts = max_vnd_accepts if idx == 0 else max(4, max_vnd_accepts // 2)
            cand = self._vnd_ils_from(sol, accepts, kicks)
            if cand.total_score > cur.total_score:
                cur = cand

        if self._phase_time_left(reserve=0.15) > 0.35:
            cand = self._deep_intensify(cur)
            if cand.total_score > cur.total_score:
                cur = cand

        self.local_search_elapsed_s = _time.time() - start
        debug_breakpoint(
            "ApexUpgradedScheduler._local_search.end",
            final_score=cur.total_score,
            attempts=self.local_search_attempts,
            improvements=self.local_search_improvements,
            elapsed=round(self.local_search_elapsed_s, 3),
        )
        return cur

    # ============================================================== entry ====
    def generate_solution(self) -> Solution:
        debug_breakpoint(
            "ApexUpgradedScheduler.generate_solution.start",
            time_limit=self.time_limit,
            population_size=self.pop_size,
        )
        self._t0 = _time.time()
        self._hard_deadline = self._t0 + self.time_limit
        planned_ls_budget = min(
            self._planned_local_search_budget(),
            max(0.0, self.time_limit * 0.35),
        )
        self._deadline = self._hard_deadline - planned_ls_budget
        if self._deadline <= self._t0 + 1.0:
            self._deadline = self._hard_deadline - max(0.25, self.time_limit * 0.10)

        self.ga_solution = None
        self.ga_score = 0
        self.ga_segments = 0
        self.ga_elapsed_s = 0.0
        self.ga_ls_solution = None
        self.ga_ls_score = 0
        self.ga_ls_segments = 0
        self.local_search_elapsed_s = 0.0
        self.local_search_attempts = 0
        self.local_search_improvements = 0
        self.local_search_improvement = 0
        self._gls_penalties = {}
        self._gls_updates = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(
                f"APEX UPGRADED  (ch={self.n_channels}  "
                f"budget={self.time_limit:.0f}s  pop={self.pop_size})"
            )
            print(f"GA phase until: {self._deadline - self._t0:.1f}s  LS reserve: {planned_ls_budget:.1f}s")
            if self._total_pref:
                print(
                    f"Prefs: {self._total_pref}  genres={self._pref_genres}  "
                    f"pen={self._pen} ratio={self._pen_ratio:.2f}"
                )
            print("=" * 60)

        raw_constr_budget = (
            self.time_limit * (1.0 - self.ga_frac) - 10.0
            if self.time_limit >= 30.0
            else self.time_limit * (1.0 - self.ga_frac)
        )
        constr_budget = min(
            max(0.2, raw_constr_budget),
            max(0.2, self._phase_time_left(reserve=1.0)),
        )
        population = self._seed(constr_budget)

        if not population:
            return Solution([], 0)

        elite_budget = min(
            22.0,
            max(0.0, self._phase_time_left(reserve=2.0) * 0.14),
        )
        if elite_budget > 1.0 and self._deadline_ok(margin=6.0):
            population = self._postprocess_elite(population, elite_budget)

        ga_budget = self._phase_time_left(reserve=0.5)
        if self.n_channels <= 100:
            min_ga_budget = 0.5
        elif self.n_channels <= 500:
            min_ga_budget = 2.0
        else:
            min_ga_budget = 5.0
        if ga_budget > min_ga_budget and len(population) >= 4:
            best = self._ga(population, ga_budget)
        else:
            best = population[0]

        self.ga_solution = best
        self.ga_score = int(best.total_score)
        self.ga_segments = len(best.scheduled_programs)
        self.ga_elapsed_s = _time.time() - self._t0

        ls_budget = min(
            planned_ls_budget,
            max(0.0, self._hard_deadline - _time.time() - 0.15),
        )
        if ls_budget > 0.05:
            best = self._local_search(best, ls_budget, start_pool=population)

        self.ga_ls_solution = best
        self.ga_ls_score = int(best.total_score)
        self.ga_ls_segments = len(best.scheduled_programs)
        self.local_search_improvement = self.ga_ls_score - self.ga_score

        if self.verbose:
            print(f"\n{'='*60}")
            print(
                f"GA RESULT     score={self.ga_score}  "
                f"segs={self.ga_segments}  "
                f"time={self.ga_elapsed_s:.1f}s"
            )
            print(
                f"GA+LS RESULT  score={self.ga_ls_score}  "
                f"segs={self.ga_ls_segments}  "
                f"improvement={self.local_search_improvement}  "
                f"ls_time={self.local_search_elapsed_s:.1f}s  "
                f"time={self._elapsed():.1f}s"
            )
            print("=" * 60 + "\n")

        debug_breakpoint(
            "ApexUpgradedScheduler.generate_solution.end",
            ga_score=self.ga_score,
            ga_ls_score=self.ga_ls_score,
            improvement=self.local_search_improvement,
        )
        return best
