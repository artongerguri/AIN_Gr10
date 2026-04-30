"""
Apex Scheduler — strongest 5-minute scheduler.

Key innovations over Ultra/Elite:
1. Continuation-chain aware: evaluates multi-segment stays on same channel,
   properly valuing the massive penalty savings from avoiding switches.
2. Adaptive ranking: density-mode for large/low-pen, value-mode for small/high-pen,
   with continuation lookahead bonus in both modes.
3. Auto-extend: after picking a segment, optionally extends the chain on the same
   channel when the continuation program is strong enough.
4. Multi-profile search: runs many strategy configurations (density, value, chain,
   pref-heavy, score-pure) and keeps the best.
5. Stronger GA: tournament-5, 50 % mutation, chain-aware destroy-rebuild.
6. Strict 5-minute budget with adaptive time allocation.
"""

import bisect
import random
import time as _time
from typing import List, Optional, Set, Tuple, Dict

from models.schedule import Schedule
from models.solution import Solution
from scheduler.beam_search_scheduler import BeamSearchScheduler


class ApexScheduler(BeamSearchScheduler):

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
        seed: Optional[int] = None,
        lookahead_limit: int = 6,
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
        self.time_limit = time_limit_seconds
        self.pop_size = population_size
        self.ga_frac = ga_time_fraction
        self.rng = random.Random(seed if seed is not None else 42)
        self._t0: float = 0.0
        self._large = self.n_channels > 100

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elite_fraction = elite_fraction
        self.stale_limit = stale_limit

        self._apex_build()

    # -------------------------------------------------------------- timing
    def _elapsed(self) -> float:
        return _time.time() - self._t0

    def _left(self, reserve: float = 2.0) -> float:
        return self.time_limit - self._elapsed() - reserve

    # --------------------------------------------------------- preprocessing
    def _apex_build(self):
        d = self.instance_data
        self._sp = d.switch_penalty
        self._tp = d.termination_penalty
        self._mc = d.max_consecutive_genre
        self._open = d.opening_time
        self._close = d.closing_time

        self._pen = self._sp + 2 * self._tp
        typval = self.avg_score_per_min * self.min_d
        self._pen_ratio = self._pen / max(1.0, typval)
        self._high_pen = self._tp > 40

        if self._pen_ratio > 0.6:
            self._fw, self._fb = 0.55, 2.5
        elif self._pen_ratio > 0.3:
            self._fw, self._fb = 0.70, 1.5
        else:
            self._fw, self._fb = 0.85, 0.8

        self._pw = sorted(
            [(p.start, p.end, p.preferred_genre, p.bonus) for p in self.prefs],
            key=lambda x: x[0],
        )
        self._pw_starts = [pw[0] for pw in self._pw]
        self._pref_genres = set(p.preferred_genre for p in self.prefs)
        self._total_pref = sum(
            pw[3] for pw in self._pw if pw[1] - pw[0] >= self.min_d
        )

        self._nxt: Dict[str, Tuple] = {}
        for ch_idx, progs in enumerate(self.ch_progs):
            for i in range(len(progs) - 1):
                self._nxt[progs[i].unique_id] = (progs[i + 1], ch_idx)

        if self.n_channels > 5000:
            self.pop_size = min(self.pop_size, 15)
        elif self.n_channels > 2000:
            self.pop_size = min(self.pop_size, 25)
        elif self.n_channels > 500:
            self.pop_size = min(self.pop_size, 40)

    # -------------------------------------------------------- pref helpers
    def _pref_for(self, genre: str, s: int, e: int) -> int:
        if genre not in self._pref_genres:
            return 0
        idx = bisect.bisect_right(self._pw_starts, s) - 1
        for i in range(max(0, idx), len(self._pw)):
            pw = self._pw[i]
            if pw[0] >= e + self.min_d:
                break
            if pw[2] != genre:
                continue
            if min(e, pw[1]) - max(s, pw[0]) >= self.min_d:
                return pw[3]
        return 0

    def _fpv(self, end: int, genre: str, streak: int) -> float:
        total = 0.0
        checked = 0
        idx = max(0, bisect.bisect_right(self._pw_starts, end) - 1)
        for i in range(idx, len(self._pw)):
            pw = self._pw[i]
            if pw[0] > end + self.min_d * 5:
                break
            if pw[1] <= end:
                continue
            checked += 1
            if checked > 3:
                break
            earliest = end + self.min_d if genre == pw[2] and streak >= self._mc else end
            if earliest <= pw[1] - self.min_d:
                total += pw[3] * 0.35
            elif earliest - self.min_d <= pw[1] - self.min_d:
                total += pw[3] * 0.08
        return total

    # ====================================================== CANDIDATES ====
    def _gen(self, t, prev_ch, prev_genre, gs, used):
        if self._large:
            return self._gen_large(t, prev_ch, prev_genre, gs, used)
        cands = self._get_candidates(t, prev_ch, prev_genre, gs, used)
        if not cands:
            return []
        if any(c[4] <= t for c in cands):
            return cands
        return []

    def _gen_large(self, t, prev_ch, prev_genre, gs, used):
        out: list = []
        close = self._close
        hp = self._high_pen

        for ch_idx in range(self.n_channels):
            prog = self._get_prog(ch_idx, t)
            if prog is None or prog.unique_id in used:
                continue
            ns = gs + 1 if prog.genre == prev_genre else 1
            if ns > self._mc:
                continue
            ch_id = self.instance_data.channels[ch_idx].channel_id
            ne = min(prog.end, close)
            if ne - t < self.min_d:
                continue

            ends: set = {ne}
            if not hp:
                me = t + self.min_d
                if me < ne:
                    ends.add(me)
            if prog.genre in self._pref_genres:
                pi = bisect.bisect_right(self._pw_starts, t) - 1
                for j in range(max(0, pi), min(len(self._pw), pi + 3)):
                    pw = self._pw[j]
                    if pw[2] == prog.genre and t < pw[1] <= ne and pw[1] - t >= self.min_d:
                        ends.add(pw[1])

            for se in ends:
                if se - t < self.min_d or se > close:
                    continue
                if not self._channel_allowed(ch_idx, t, se):
                    continue
                sc = self._calc_score(prog, ch_idx, t, se, prev_ch)
                if sc > -999999:
                    out.append((sc, ch_idx, ch_id, prog, t, se))

        if out:
            si = bisect.bisect_right(self.times, t)
            la = 0
            for i in range(si, len(self.times)):
                ft = self.times[i]
                if ft >= close or ft > t + self.min_d * self.lookahead_limit:
                    break
                la += 1
                if la > self.lookahead_limit:
                    break
                for prog, ch_idx in self.starts_at.get(ft, []):
                    if prog.unique_id in used:
                        continue
                    ns = gs + 1 if prog.genre == prev_genre else 1
                    if ns > self._mc:
                        continue
                    ch_id = self.instance_data.channels[ch_idx].channel_id
                    ne = min(prog.end, close)
                    if ne - ft < self.min_d:
                        continue
                    la_ends = {ne}
                    if not hp:
                        lme = ft + self.min_d
                        if lme < ne:
                            la_ends.add(lme)
                    for se in la_ends:
                        if se - ft < self.min_d or se > close:
                            continue
                        if not self._channel_allowed(ch_idx, ft, se):
                            continue
                        sc = self._calc_score(prog, ch_idx, ft, se, prev_ch)
                        if sc > -999999:
                            out.append((sc, ch_idx, ch_id, prog, ft, se))
        return out

    # ========================================================== RANKING ====
    def _rank(self, cands, t, prev_ch, prev_genre, gs, strategy, used):
        scored = []
        aspm = self.avg_score_per_min
        close = self._close

        force_val = strategy.startswith("v_")
        strat = strategy[2:] if force_val else strategy
        use_density = self._large and not force_val

        for c in cands:
            seg_sc, ch_idx, ch_id, prog, ss, se = c
            dt = max(1, se - t)
            remaining = close - se

            ns = gs + 1 if prog.genre == prev_genre else 1
            fpv = self._fpv(se, prog.genre, ns)

            # ---------- base + future ----------
            if use_density:
                base = seg_sc / dt * self.min_d
                future = remaining * aspm * 0.03
            else:
                base = seg_sc + remaining * aspm * self._fw
                future = 0.0

            # ---------- full-program bonus (value mode) ----------
            fb = 0.0
            if not use_density:
                if ss <= prog.start and se >= min(prog.end, close):
                    fb = self._tp * self._fb
                nxi = self._nxt.get(prog.unique_id)
                if nxi is not None and se >= prog.end:
                    if nxi[0].start <= se:
                        fb += nxi[0].score * 0.12

            # ---------- continuation-lookahead bonus ----------
            cont_la = 0.0
            if se >= min(prog.end, close):
                nxi = self._nxt.get(prog.unique_id)
                if nxi is not None:
                    np_prog, np_ch = nxi
                    np_ok = (
                        np_prog.unique_id not in used
                        and np_prog.start <= se
                        and min(np_prog.end, close) - se >= self.min_d
                    )
                    if np_ok:
                        saved = self._sp + self._tp
                        np_gs = ns + 1 if np_prog.genre == prog.genre else 1
                        if np_gs <= self._mc:
                            cont_la = saved * 0.5
                            nxi2 = self._nxt.get(np_prog.unique_id)
                            if nxi2 is not None:
                                np2 = nxi2[0]
                                np2_gs = np_gs + 1 if np2.genre == np_prog.genre else 1
                                if (
                                    np2.unique_id not in used
                                    and np2.start <= min(np_prog.end, close)
                                    and np2_gs <= self._mc
                                ):
                                    cont_la += saved * 0.25

            # ---------- strategy-specific composite ----------
            if strat == "max_pref":
                pb = self._pref_for(prog.genre, ss, se)
                composite = base + fb + fpv * 1.5 + pb * 0.4 + future
            elif strat == "chain":
                composite = base + fb + cont_la * 2.0 + fpv + future
            elif strat == "continuation":
                cx = 0.0
                nxi = self._nxt.get(prog.unique_id)
                if nxi and se >= prog.end:
                    cx = (self._sp + 2 * self._tp) * 0.3
                composite = base + fb + cx + future * 0.5 + fpv * 0.5 + cont_la
            elif strat == "balanced":
                cx = 0.0
                nxi = self._nxt.get(prog.unique_id)
                if nxi and se >= prog.end:
                    cx = (self._sp + self._tp) * 0.15
                composite = base + fb + fpv + cx + future
            else:
                composite = base + fb + future + fpv

            scored.append((composite, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ======================================================= CONSTRUCTION ==
    def _construct(
        self,
        strategy: str = "density",
        alpha: float = 0.0,
        mc_override=None,
        prefix=None,
        time_limit=None,
        extend: bool = False,
    ) -> Solution:
        close = self._close
        mc = mc_override if mc_override is not None else self._mc

        if prefix and len(prefix) > 0:
            sched = list(prefix)
            used = {s.unique_program_id for s in sched}
            total = sum(s.fitness for s in sched)
            last = sched[-1]
            t = last.end
            prev_ch = last.channel_id
            pi = self.prog_by_id.get(last.unique_program_id)
            prev_genre = pi[0].genre if pi else ""
            gs = self._streak(sched)
        else:
            sched: List[Schedule] = []
            used: Set[str] = set()
            total = 0
            t = self._open
            prev_ch: Optional[int] = None
            prev_genre = ""
            gs = 0

        deadline = _time.time() + time_limit if time_limit else float("inf")
        iters = 0

        while t < close and iters < 20000:
            if time_limit and _time.time() > deadline:
                break
            iters += 1

            cands = self._gen(t, prev_ch, prev_genre, gs, used)
            if mc < self._mc:
                cands = [
                    c for c in cands
                    if c[3].genre != prev_genre or gs + 1 <= mc
                ]
                if not cands:
                    cands = self._gen(t, prev_ch, prev_genre, gs, used)
            if not cands:
                idx = bisect.bisect_right(self.times, t)
                if idx < len(self.times) and self.times[idx] < close:
                    t = self.times[idx]
                    continue
                break

            ranked = self._rank(cands, t, prev_ch, prev_genre, gs, strategy, used)

            if alpha > 0 and len(ranked) > 1:
                bv, wv = ranked[0][0], ranked[-1][0]
                thr = bv - alpha * (bv - wv) if bv > wv else bv
                rcl = [r for r in ranked if r[0] >= thr]
                pick = self.rng.choice(rcl)[1]
            else:
                pick = ranked[0][1]

            seg_sc, ch_idx, ch_id, prog, ss, se = pick
            sched.append(
                Schedule(prog.program_id, ch_id, ss, se, seg_sc, prog.unique_id)
            )
            total += seg_sc
            used.add(prog.unique_id)
            gs = (gs + 1) if prog.genre == prev_genre else 1
            prev_genre = prog.genre
            prev_ch = ch_id
            t = se

            if extend and se >= min(prog.end, close):
                t, total, gs, prev_genre = self._auto_extend(
                    prog, ch_idx, ch_id, se, prev_genre, gs, used, sched,
                    total, close, mc,
                )

        return Solution(sched, total)

    def _auto_extend(self, prog, ch_idx, ch_id, t, prev_genre, gs, used,
                     sched, total, close, mc):
        threshold = max(0, self.avg_score_per_min * self.min_d - self._pen) * 0.3
        while t < close:
            nxi = self._nxt.get(prog.unique_id)
            if not nxi:
                break
            np_prog, np_ch = nxi
            if np_prog.unique_id in used or np_prog.start > t:
                break
            ne = min(np_prog.end, close)
            if ne - t < self.min_d:
                break
            ns = gs + 1 if np_prog.genre == prev_genre else 1
            if ns > mc:
                break
            if not self._channel_allowed(np_ch, t, ne):
                break
            sc = self._calc_score(np_prog, np_ch, t, ne, ch_id)
            if sc < threshold:
                break
            sched.append(
                Schedule(np_prog.program_id, ch_id, t, ne, sc, np_prog.unique_id)
            )
            total += sc
            used.add(np_prog.unique_id)
            gs = ns
            prev_genre = np_prog.genre
            t = ne
            prog = np_prog
        return t, total, gs, prev_genre

    def _streak(self, sched: List[Schedule]) -> int:
        if not sched:
            return 0
        pi = self.prog_by_id.get(sched[-1].unique_program_id)
        if not pi:
            return 1
        genre = pi[0].genre
        s = 0
        for x in reversed(sched):
            info = self.prog_by_id.get(x.unique_program_id)
            if info and info[0].genre == genre:
                s += 1
            else:
                break
        return s

    # =================================================== POPULATION SEED ===
    def _seed(self, budget: float) -> List[Solution]:
        deadline = _time.time() + budget
        pool: List[Solution] = []

        # Beam seed for small instances
        if self.n_channels <= 20:
            bw, li = 200, 50
        elif self.n_channels <= 50:
            bw, li = 150, 30
        elif self.n_channels <= 100:
            bw, li = 60, 15
        else:
            bw, li = 0, 0

        if bw > 0 and _time.time() < deadline:
            orig = self.beam_width
            self.beam_width = bw
            bs = self._beam_search_core()
            pool.append(bs)
            self.beam_width = orig
            if self.verbose:
                print(f"  beam: {bs.total_score} ({self._elapsed():.1f}s)")

        core_strats = ["density", "balanced", "continuation"]
        extra_strats = ["max_pref"]
        if self._large:
            extra_strats += ["v_density", "v_balanced", "v_continuation"]

        configs_core = [(s, 0.0, None, False) for s in core_strats]
        configs_core.append(("chain", 0.0, None, True))
        configs_extra = [(s, 0.0, None, False) for s in extra_strats]
        if self._pref_genres:
            configs_core += [(s, 0.0, 1, False) for s in core_strats[:2]]
            configs_core.append(("chain", 0.0, 1, True))
            configs_extra += [(s, 0.0, 1, False) for s in extra_strats]

        t_per = 0.0
        for strat, alpha, mc, ext in configs_core:
            if _time.time() >= deadline:
                break
            tc = _time.time()
            sol = self._construct(strategy=strat, alpha=alpha,
                                  mc_override=mc, extend=ext)
            t_per = max(t_per, _time.time() - tc)
            pool.append(sol)
            lbl = strat + (f" mc={mc}" if mc is not None else "") + (" ext" if ext else "")
            if self.verbose:
                print(f"  {lbl}: {sol.total_score} ({self._elapsed():.1f}s)")

        remaining_budget = deadline - _time.time()
        extra_budget = remaining_budget * 0.3
        if t_per > 0 and extra_budget / max(0.01, t_per) >= 2:
            et0 = _time.time()
            for strat, alpha, mc, ext in configs_extra:
                if _time.time() >= deadline or _time.time() - et0 > extra_budget:
                    break
                sol = self._construct(strategy=strat, alpha=alpha,
                                      mc_override=mc, extend=ext)
                pool.append(sol)
                lbl = strat + (f" mc={mc}" if mc is not None else "") + (" ext" if ext else "")
                if self.verbose:
                    print(f"  {lbl}: {sol.total_score} ({self._elapsed():.1f}s)")

        alphas = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
        grasp_strats = ["density", "balanced", "continuation", "chain"]
        if self._large and t_per < 2.0:
            grasp_strats += ["v_density", "v_balanced"]
        mc_opts = [None, 1] if self._pref_genres else [None]

        gi = 0
        while _time.time() < deadline and len(pool) < self.pop_size * 3:
            a = alphas[gi % len(alphas)]
            s = grasp_strats[(gi // len(alphas)) % len(grasp_strats)]
            mc = mc_opts[(gi // (len(alphas) * len(grasp_strats))) % len(mc_opts)]
            ext = s == "chain"
            sol = self._construct(strategy=s, alpha=a, mc_override=mc, extend=ext)
            pool.append(sol)
            gi += 1

        pool.sort(key=lambda s: s.total_score, reverse=True)
        kept: List[Solution] = []
        seen: set = set()
        for sol in pool:
            sig = frozenset(s.unique_program_id for s in sol.scheduled_programs)
            if sig not in seen:
                seen.add(sig)
                kept.append(sol)
            if len(kept) >= self.pop_size:
                break
        for sol in pool:
            if len(kept) >= self.pop_size:
                break
            if sol not in kept:
                kept.append(sol)

        if self.verbose and kept:
            print(
                f"  Pool: {len(kept)} of {len(pool)} "
                f"[best={kept[0].total_score} worst={kept[-1].total_score} "
                f"t={self._elapsed():.1f}s]"
            )
        return kept

    # ======================================================== GA OPERATORS ==
    def _tournament(self, pop: List[Solution], k: int = 5) -> Solution:
        return max(
            self.rng.sample(pop, min(k, len(pop))),
            key=lambda s: s.total_score,
        )

    def _crossover(self, p1: Solution, p2: Solution) -> Solution:
        s1 = p1.scheduled_programs
        s2 = p2.scheduled_programs
        if len(s1) < 2 or len(s2) < 2:
            return p1 if p1.total_score >= p2.total_score else p2

        cut = self.rng.randint(1, len(s1) - 1)
        prefix = s1[:cut]
        cut_time = prefix[-1].end
        used = {s.unique_program_id for s in prefix}
        suffix = [
            s for s in s2
            if s.start >= cut_time and s.unique_program_id not in used
        ]
        return self._repair(prefix + suffix)

    def _mutate(self, sol: Solution) -> Solution:
        progs = sol.scheduled_programs
        if len(progs) < 2:
            strat = self.rng.choice(["density", "chain", "balanced"])
            return self._construct(strategy=strat, alpha=0.15,
                                   extend=self.rng.random() < 0.5)

        r = self.rng.random()
        if r < 0.40:
            idx = self._weak_region(progs)
        elif r < 0.70:
            idx = min(range(len(progs)), key=lambda i: progs[i].fitness)
        else:
            idx = self.rng.randint(0, len(progs) - 1)

        prefix = progs[:idx] if idx > 0 else None
        strats = ["density", "balanced", "chain", "continuation"]
        if self._pref_genres:
            strats.append("max_pref")
        return self._construct(
            strategy=self.rng.choice(strats),
            alpha=self.rng.uniform(0.0, 0.20),
            mc_override=self.rng.choice([None, 1]) if self._pref_genres else None,
            prefix=prefix,
            extend=self.rng.random() < 0.5,
        )

    def _weak_region(self, progs: List[Schedule]) -> int:
        w = min(3, len(progs))
        worst_avg = float("inf")
        worst_idx = 0
        for i in range(len(progs) - w + 1):
            avg = sum(progs[j].fitness for j in range(i, i + w)) / w
            if avg < worst_avg:
                worst_avg = avg
                worst_idx = i
        return worst_idx

    def _repair(self, segments: List[Schedule]) -> Solution:
        if not segments:
            return self._construct(strategy="density", alpha=0.1)
        segments = sorted(segments, key=lambda s: s.start)
        clean: List[Schedule] = []
        used: Set[str] = set()
        pe = self._open
        pc: Optional[int] = None
        pg = ""
        gs = 0
        for s in segments:
            if s.start < pe or s.unique_program_id in used:
                continue
            pi = self.prog_by_id.get(s.unique_program_id)
            if not pi:
                continue
            prog, ch_idx = pi
            if s.end - s.start < self.min_d or s.end > self._close:
                continue
            if not self._channel_allowed(ch_idx, s.start, s.end):
                continue
            ns = (gs + 1) if prog.genre == pg else 1
            if ns > self._mc:
                continue
            ch_id = self.instance_data.channels[ch_idx].channel_id
            sc = self._calc_score(prog, ch_idx, s.start, s.end, pc)
            if sc <= -999999:
                continue
            clean.append(
                Schedule(s.program_id, ch_id, s.start, s.end, sc,
                         s.unique_program_id)
            )
            used.add(s.unique_program_id)
            pe = s.end
            pc = ch_id
            gs = ns
            pg = prog.genre
        return self._construct(
            strategy="density", alpha=0.0,
            prefix=clean if clean else None,
        )

    # ============================================================ GA LOOP ==
    def _ga(self, pop: List[Solution], budget: float) -> Solution:
        deadline = _time.time() + budget
        elite_n = max(2, int(len(pop) * self.elite_fraction))
        best = pop[0]
        stale = 0
        gen = 0

        if self._large:
            default_cx, default_mut, default_stale, default_tk = 0.55, 0.50, 50, 5
        else:
            default_cx, default_mut, default_stale, default_tk = 0.65, 0.40, 60, 3

        cx_rate = self.crossover_rate if self.crossover_rate is not None else default_cx
        mut_rate = self.mutation_rate if self.mutation_rate is not None else default_mut
        stale_lim = self.stale_limit if self.stale_limit is not None else default_stale
        tk = self.tournament_k if self.tournament_k is not None else default_tk

        while _time.time() < deadline and stale < stale_lim:
            gen += 1
            nxt: List[Solution] = list(pop[:elite_n])

            while len(nxt) < len(pop) and _time.time() < deadline:
                p1 = self._tournament(pop, tk)
                p2 = self._tournament(pop, tk)
                if self.rng.random() < cx_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = max(p1, p2, key=lambda s: s.total_score)
                if self.rng.random() < mut_rate:
                    child = self._mutate(child)
                nxt.append(child)

            nxt.sort(key=lambda s: s.total_score, reverse=True)
            pop = nxt[: self.pop_size]

            if pop[0].total_score > best.total_score:
                best = pop[0]
                stale = 0
                if self.verbose:
                    print(f"    GA gen {gen}: {best.total_score}")
            else:
                stale += 1

        if self.verbose:
            print(f"  GA: {gen} gens, best={best.total_score}")
        return best

    # ======================================================= ENTRY POINT ==
    def generate_solution(self) -> Solution:
        self._t0 = _time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(
                f"APEX SCHEDULER  (ch={self.n_channels}  "
                f"budget={self.time_limit:.0f}s  pop={self.pop_size})"
            )
            if self._total_pref:
                print(
                    f"Prefs: {self._total_pref}  genres={self._pref_genres}  "
                    f"pen={self._pen} ratio={self._pen_ratio:.2f}"
                )
            print("=" * 60)

        constr_budget = max(5.0, self.time_limit * (1.0 - self.ga_frac) - 2.0)
        population = self._seed(constr_budget)

        if not population:
            return Solution([], 0)

        ga_budget = self._left(reserve=3.0)
        if ga_budget > 8.0 and len(population) >= 4:
            best = self._ga(population, ga_budget)
        else:
            best = population[0]

        if self.verbose:
            print(f"\n{'='*60}")
            print(
                f"RESULT  score={best.total_score}  "
                f"segs={len(best.scheduled_programs)}  "
                f"time={self._elapsed():.1f}s"
            )
            print("=" * 60 + "\n")

        return best
