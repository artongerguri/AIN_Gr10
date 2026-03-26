from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import bisect
import heapq

from models.instance_data import InstanceData
from models.solution import Solution
from models.schedule import Schedule
from models.program import Program
from utils.utils import Utils


class BeamSearchScheduler:

    def __init__(self, instance_data: InstanceData, 
                 beam_width: int = 50,
                 lookahead_limit: int = 4,
                 density_percentile: int = 25,
                 verbose: bool = True):
        self.instance_data = instance_data
        self.beam_width = beam_width
        self.lookahead_limit = lookahead_limit
        self.density_percentile = density_percentile
        self.verbose = verbose
        self.min_d = instance_data.min_duration
        
        self._preprocess()
    
    def _preprocess(self):
        """Build all necessary indices."""
        self.n_channels = len(self.instance_data.channels)
        
        # Programs sorted by start time per channel
        self.ch_progs: List[List[Program]] = []
        self.ch_starts: List[List[int]] = []
        
        # Program lookup
        self.prog_by_id: Dict[str, Tuple[Program, int]] = {}
        
        # Start time index for fast lookahead
        self.starts_at = defaultdict(list)
        
        # All decision points (program boundaries)
        all_times = set()
        all_times.add(self.instance_data.opening_time)
        
        for ch_idx, channel in enumerate(self.instance_data.channels):
            progs = sorted(channel.programs, key=lambda p: p.start)
            self.ch_progs.append(progs)
            self.ch_starts.append([p.start for p in progs])
            
            for prog in progs:
                all_times.add(prog.start)
                all_times.add(prog.end)
                self.prog_by_id[prog.unique_id] = (prog, ch_idx)
                self.starts_at[prog.start].append((prog, ch_idx))
        
        # Filter to valid range
        self.times = sorted([t for t in all_times 
                            if self.instance_data.opening_time <= t <= self.instance_data.closing_time])
        
        # CRITICAL: Add priority block boundaries as decision points
        # This enables late starts after a priority block ends
        for block in self.instance_data.priority_blocks:
            if block.start not in self.times:
                self.times.append(block.start)
            if block.end not in self.times:
                self.times.append(block.end)
        self.times = sorted(set(self.times))
        
        # Priority block index: time -> set of allowed channel_ids
        self.priority_at: Dict[int, Set[int]] = {}
        for block in self.instance_data.priority_blocks:
            allowed = set(block.allowed_channels)
            for t in range(block.start, block.end):
                if t not in self.priority_at:
                    self.priority_at[t] = allowed.copy()
                else:
                    self.priority_at[t] &= allowed
        
        # Precompute forbidden intervals for O(1) checks
        self.forbidden_prefix = []
        self.has_priority_blocks = bool(self.instance_data.priority_blocks)
        
        if self.has_priority_blocks:
            max_t = self.instance_data.closing_time + 100  # Buffer
            for ch_idx, channel in enumerate(self.instance_data.channels):
                ch_id = channel.channel_id
                # Build boolean array: 1 if forbidden, 0 if allowed
                is_forbidden = [0] * max_t
                for t, allowed in self.priority_at.items():
                    if t < max_t and ch_id not in allowed:
                        is_forbidden[t] = 1
                
                # Build prefix sum
                prefix = [0] * (max_t + 1)
                curr = 0
                for t in range(max_t):
                    curr += is_forbidden[t]
                    prefix[t+1] = curr
                self.forbidden_prefix.append(prefix)
        
        # Time preference index for faster lookup
        self.prefs = self.instance_data.time_preferences

        # Calculate average score per minute for heuristics
        # IMPROVED: Use top 25% of programs to get a more realistic "good" density
        # This prevents the scheduler from wasting time on low-value gaps
        densities = []
        for ch in self.instance_data.channels:
            for p in ch.programs:
                dur = p.end - p.start
                if dur > 0:
                    densities.append(p.score / dur)
        
        densities.sort(reverse=True)
        if densities:
            # Take top N% based on density_percentile
            # Lower percentile = More optimistic (expects higher scores)
            # Higher percentile = More conservative
            top_n = max(1, int(len(densities) * (self.density_percentile / 100.0)))
            self.avg_score_per_min = sum(densities[:top_n]) / top_n
        else:
            self.avg_score_per_min = 1.0
            
        if self.verbose:
            print(f"Average score density (top {self.density_percentile}%): {self.avg_score_per_min:.4f} pts/min")
    
    def _get_prog(self, ch_idx: int, time: int) -> Optional[Program]:
        """Get program at time on channel (binary search)."""
        idx = bisect.bisect_right(self.ch_starts[ch_idx], time) - 1
        if 0 <= idx < len(self.ch_progs[ch_idx]):
            p = self.ch_progs[ch_idx][idx]
            if p.start <= time < p.end:
                return p
        return None
    
    def _channel_allowed(self, ch_idx: int, start: int, end: int) -> bool:
        """Check if channel is allowed for entire duration (O(1) using prefix sums)."""
        if not self.has_priority_blocks:
            return True
            
        # Check prefix sum
        # Count of forbidden minutes in [start, end) is prefix[end] - prefix[start]
        prefix = self.forbidden_prefix[ch_idx]
        max_t = len(prefix) - 1
        
        # Clamp to valid range
        s = min(start, max_t)
        e = min(end, max_t)
        
        if s >= e:
            return True
            
        count = prefix[e] - prefix[s]
        return count == 0
    
    def _calc_score(self, prog: Program, ch_idx: int, 
                    seg_start: int, seg_end: int,
                    prev_ch_id: Optional[int]) -> int:
        """
        Calculate score for scheduling a segment.
        
        Args:
            prog: The program being watched
            ch_idx: Channel index
            seg_start: When we start watching (can be after prog.start)
            seg_end: When we stop watching (can be before prog.end)
            prev_ch_id: Previous channel (for switch penalty)
        """
        duration = seg_end - seg_start
        if duration < self.min_d:
            return -999999
        
        channel = self.instance_data.channels[ch_idx]
        
        # Base score - full program score regardless of partial viewing
        score = prog.score
        
        # Time preference bonus
        # Per PDF: "the program must fall within the preferred interval with at least D"
        # This means we check if the SCHEDULED SEGMENT overlaps the preference by >= D
        for pref in self.prefs:
            if prog.genre == pref.preferred_genre:
                # Check overlap between our SEGMENT and the preference window
                ov_start = max(seg_start, pref.start)
                ov_end = min(seg_end, pref.end)
                if ov_end - ov_start >= self.min_d:
                    score += pref.bonus
                    break  # Only one bonus per preference match
        
        # Switch penalty
        if prev_ch_id is not None and prev_ch_id != channel.channel_id:
            score -= self.instance_data.switch_penalty
            
        # Termination penalty (Late Start / Early Stop)
        # Late Start: If we start watching after the program's official start
        if seg_start > prog.start:
            score -= self.instance_data.termination_penalty
            
        if seg_end < prog.end:
            score -= self.instance_data.termination_penalty
        
        return score
    
    def _get_candidates(self, time: int, prev_ch_id: Optional[int],
                        prev_genre: str, genre_streak: int,
                        used_progs: Set[str]) -> List[Tuple[int, int, int, Program, int, int]]:
        """
        Get all valid segment candidates starting from current time.
        
        KEY INSIGHT: We can join a program that's already in progress (late start)!
        The program just needs to still be running at 'time'.
        
        Returns: List of (score, ch_idx, ch_id, prog, seg_start, seg_end)
        """
        candidates = []
        closing = self.instance_data.closing_time
        
        for ch_idx in range(self.n_channels):
            channel = self.instance_data.channels[ch_idx]
            ch_id = channel.channel_id
            
            # Get program running at current time (could have started earlier = late start)
            prog = self._get_prog(ch_idx, time)
            if not prog:
                continue
            
            # Skip if we already used this exact program
            if prog.unique_id in used_progs:
                continue
            
            # Genre constraint
            new_streak = 1 if prog.genre != prev_genre else genre_streak + 1
            if new_streak > self.instance_data.max_consecutive_genre:
                continue
            
            # The segment starts at current time (late start if time > prog.start)
            seg_start = time
            
            # Try different end times
            end_options = set()
            
            # Option 1: Natural program end
            nat_end = min(prog.end, closing)
            if nat_end - seg_start >= self.min_d:
                end_options.add(nat_end)
            
            # Option 2: Early stops at program boundaries
            # Optimize: Use bisect to find relevant times
            start_idx = bisect.bisect_right(self.times, seg_start + self.min_d)
            end_idx = bisect.bisect_left(self.times, nat_end)
            
            for i in range(start_idx, end_idx + 1):
                if i >= len(self.times):
                    break
                t = self.times[i]
                if t > nat_end:
                    break
                end_options.add(t)
            
            # Option 3: End at exact min_duration
            min_end = seg_start + self.min_d
            if min_end <= nat_end:
                end_options.add(min_end)
            
            for seg_end in sorted(end_options):
                if seg_end > closing:
                    continue
                
                # Check priority blocks for the segment
                if not self._channel_allowed(ch_idx, seg_start, seg_end):
                    continue
                
                score = self._calc_score(prog, ch_idx, seg_start, seg_end, prev_ch_id)
                if score > -999999:
                    candidates.append((score, ch_idx, ch_id, prog, seg_start, seg_end))
        
        # Also try looking ahead for future programs that might offer better value
        # This helps when current time has no good options but a program starts soon
        # Improved lookahead: Check actual future time points instead of fixed offsets
        start_idx = bisect.bisect_right(self.times, time)
        lookahead_count = 0
        
        for i in range(start_idx, len(self.times)):
            future_time = self.times[i]
            if future_time >= closing:
                break
            # Don't look too far ahead (performance)
            if future_time > time + self.min_d * self.lookahead_limit:
                break
                
            lookahead_count += 1
            if lookahead_count > self.lookahead_limit: # Limit lookahead checks
                break
            
            # Optimized: Use starts_at index instead of iterating all channels
            for prog, ch_idx in self.starts_at.get(future_time, []):
                channel = self.instance_data.channels[ch_idx]
                ch_id = channel.channel_id
                
                if prog.unique_id in used_progs:
                    continue
                
                # Only consider if this is a program START (not late join)
                # (Implicitly true because we used starts_at)
                
                new_streak = 1 if prog.genre != prev_genre else genre_streak + 1
                if new_streak > self.instance_data.max_consecutive_genre:
                    continue
                
                nat_end = min(prog.end, closing)
                if nat_end - future_time < self.min_d:
                    continue
                
                if not self._channel_allowed(ch_idx, future_time, nat_end):
                    continue
                
                # Use future_time as start (with a small penalty for waiting)
                score = self._calc_score(prog, ch_idx, future_time, nat_end, prev_ch_id)
                if score > -999999:
                    candidates.append((score, ch_idx, ch_id, prog, future_time, nat_end))
        
        return candidates
    
    def _beam_search_core(self) -> Solution:
        """
        Core beam search algorithm.
        Deterministic version.
        """
        opening = self.instance_data.opening_time
        closing = self.instance_data.closing_time
        
        # State: (score, time, prev_ch_id, prev_genre, genre_streak, schedule_tuple, used_set_tuple)
        initial = (0, opening, None, "", 0, tuple(), frozenset())
        beam = [initial]
        
        best_solution = (0, [])
        
        iterations = 0
        max_iterations = 5000  # Safety limit
        
        while beam and iterations < max_iterations:
            iterations += 1
            next_beam = []
            
            for state in beam:
                score, time, prev_ch, prev_genre, g_streak, sched_tuple, used = state
                
                if time >= closing:
                    if score > best_solution[0]:
                        best_solution = (score, list(sched_tuple))
                    continue
                
                candidates = self._get_candidates(time, prev_ch, prev_genre, g_streak, used)
                
                if not candidates:
                    # Jump to next decision time
                    idx = bisect.bisect_right(self.times, time)
                    if idx < len(self.times) and self.times[idx] < closing:
                        next_time = self.times[idx]
                        next_beam.append((score, next_time, prev_ch, prev_genre, g_streak, sched_tuple, used))
                    else:
                        # Terminal
                        if score > best_solution[0]:
                            best_solution = (score, list(sched_tuple))
                    continue
                
                # Sort by score density heuristic: score + potential of remaining time
                # This prefers candidates that give high score for less time usage
                candidates.sort(key=lambda x: x[0] + (closing - x[5]) * self.avg_score_per_min, reverse=True)
                
                # Take top candidates
                take_n = max(3, self.beam_width // len(beam) if len(beam) > 0 else self.beam_width)
                
                for i, (seg_score, ch_idx, ch_id, prog, seg_start, seg_end) in enumerate(candidates[:take_n]):
                    new_sched = sched_tuple + ((prog.unique_id, ch_id, seg_start, seg_end, seg_score),)
                    new_used = used | {prog.unique_id}
                    new_streak = 1 if prog.genre != prev_genre else g_streak + 1
                    
                    next_beam.append((
                        score + seg_score,
                        seg_end,
                        ch_id,
                        prog.genre,
                        new_streak,
                        new_sched,
                        new_used
                    ))
            if not next_beam:
                break
            
            # Keep best states
            # Sort by heuristic: accumulated_score + potential_future_score
            next_beam.sort(key=lambda x: x[0] + (closing - x[1]) * self.avg_score_per_min, reverse=True)
            
            # Deduplicate by (time, prev_ch, genre_streak)
            seen = set()
            unique_beam = []
            for state in next_beam:
                key = (state[1], state[2], state[4])  # time, prev_ch, g_streak
                if key not in seen:
                    seen.add(key)
                    unique_beam.append(state)
                if len(unique_beam) >= self.beam_width:
                    break
            
            beam = unique_beam
        
        # Convert to Solution
        scheduled = []
        for item in best_solution[1]:
            prog_id, ch_id, start, end, seg_score = item
            prog_info = self.prog_by_id.get(prog_id)
            if prog_info:
                prog, _ = prog_info
                scheduled.append(Schedule(
                    program_id=prog.program_id,
                    channel_id=ch_id,
                    start=start,
                    end=end,
                    fitness=seg_score,
                    unique_program_id=prog_id
                ))
        
        return Solution(scheduled, best_solution[0])
    
    def _local_search(self, sol: Solution, max_iter: int = 50) -> Solution:
        """Improve solution with local search."""
        if not sol.scheduled_programs:
            return sol
        
        best = sol.scheduled_programs[:]
        best_score = sol.total_score
        
        for _ in range(max_iter):
            improved = False
            
            # Try removing each program and re-fill greedily
            for i in range(len(best)):
                prefix = best[:i]
                
                # Get state from prefix
                if prefix:
                    last = prefix[-1]
                    time = last.end
                    prev_ch = last.channel_id
                    prog_info = self.prog_by_id.get(last.unique_program_id)
                    prev_genre = prog_info[0].genre if prog_info else ""
                    g_streak = sum(1 for j in range(len(prefix)-1, -1, -1) 
                                   if self.prog_by_id.get(prefix[j].unique_program_id, (None,))[0]
                                   and self.prog_by_id[prefix[j].unique_program_id][0].genre == prev_genre)
                else:
                    time = self.instance_data.opening_time
                    prev_ch = None
                    prev_genre = ""
                    g_streak = 0
                
                # Greedy fill from here
                new_sched = prefix[:]
                new_score = sum(s.fitness for s in prefix)
                used = {s.unique_program_id for s in prefix}
                
                while time < self.instance_data.closing_time:
                    candidates = self._get_candidates(time, prev_ch, prev_genre, g_streak, used)
                    
                    if not candidates:
                        idx = bisect.bisect_right(self.times, time)
                        if idx < len(self.times):
                            time = self.times[idx]
                        else:
                            break
                        continue
                    
                    # Optimized: Use score density heuristic
                    candidates.sort(key=lambda x: x[0] + (self.instance_data.closing_time - x[5]) * self.avg_score_per_min, reverse=True)
                    seg_score, ch_idx, ch_id, prog, seg_start, seg_end = candidates[0]
                    
                    new_sched.append(Schedule(
                        program_id=prog.program_id,
                        channel_id=ch_id,
                        start=seg_start,
                        end=seg_end,
                        fitness=seg_score,
                        unique_program_id=prog.unique_id
                    ))
                    new_score += seg_score
                    used.add(prog.unique_id)
                    
                    if prog.genre == prev_genre:
                        g_streak += 1
                    else:
                        g_streak = 1
                        prev_genre = prog.genre
                    
                    prev_ch = ch_id
                    time = seg_end
                
                if new_score > best_score:
                    best = new_sched
                    best_score = new_score
                    improved = True
                    break
            
            if not improved:
                break
        
        return Solution(best, best_score)
    
    def generate_solution(self) -> Solution:
        """Generate the maximum score solution."""
        # Adaptive parameters for large instances
        if self.n_channels > 50:
            # Optimized: Set to 500 for good balance of speed and score
            if self.beam_width < 500:
                if self.verbose:
                    print(f"Large instance detected ({self.n_channels} channels). Setting beam width to 500 for speed/quality balance.")
                self.beam_width = 500
        if self.verbose:
            print(f"\n{'='*70}")
            print("MAX SCORE SCHEDULER")
            print(f"Channels: {self.n_channels}, Beam: {self.beam_width}")
            print(f"{'='*70}\n")
        
        # Strategy: Beam search (deterministic)
        if self.verbose:
            print("Running Beam Search...")
        sol = self._beam_search_core()
        
        # Always run local search, but with fewer iterations for large instances
        iter_limit = 50 if self.n_channels <= 50 else 20
        sol = self._local_search(sol, max_iter=iter_limit)
        
        if self.verbose:
            print(f"  Score: {sol.total_score}")
            print(f"\n{'='*70}")
            print(f"BEST: Score={sol.total_score}, Programs={len(sol.scheduled_programs)}")
            print(f"{'='*70}\n")
        
        return sol
