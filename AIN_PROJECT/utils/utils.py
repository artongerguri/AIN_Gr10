from typing import List, Optional

from models.channel import Channel
from models.instance_data import InstanceData
 


class Utils:
    _current_instance: InstanceData | None = None
    # Caches built from the current instance
    _channel_to_index: dict[int, int] | None = None
    _unique_id_to_program: dict[str, object] | None = None
    _channel_to_sorted_programs: dict[int, List[object]] | None = None
    _channel_to_program_starts: dict[int, List[int]] | None = None

    @staticmethod
    def _build_caches():
        if Utils._current_instance is None:
            return
        instance = Utils._current_instance
        # channel -> index cache (use id(channel) for identity)
        Utils._channel_to_index = {id(ch): idx for idx, ch in enumerate(instance.channels)}

        # unique_id -> Program cache
        uid_map: dict[str, object] = {}
        # per-channel sorted programs and start arrays
        ch_to_sorted: dict[int, List[object]] = {}
        ch_to_starts: dict[int, List[int]] = {}

        for ch in instance.channels:
            # sort programs by start time
            sorted_programs = sorted(ch.programs, key=lambda p: p.start)
            ch_to_sorted[id(ch)] = sorted_programs
            ch_to_starts[id(ch)] = [p.start for p in sorted_programs]
            for p in sorted_programs:
                if getattr(p, "unique_id", None) is not None:
                    uid_map[p.unique_id] = p

        Utils._unique_id_to_program = uid_map
        Utils._channel_to_sorted_programs = ch_to_sorted
        Utils._channel_to_program_starts = ch_to_starts

    @staticmethod
    def set_current_instance(instance_data: InstanceData):
        """Set the current instance globally for utils lookups."""
        Utils._current_instance = instance_data
        # rebuild caches
        Utils._build_caches()

    @staticmethod
    def get_channel_program_by_time(channel: Channel, time: int):
        # prefer cached binary search when current instance is set
        if Utils._current_instance is not None and Utils._channel_to_sorted_programs is not None:
            ch_id = id(channel)
            programs = Utils._channel_to_sorted_programs.get(ch_id)
            starts = Utils._channel_to_program_starts.get(ch_id) if Utils._channel_to_program_starts else None
            if programs and starts:
                # binary search: rightmost start <= time
                lo, hi = 0, len(starts) - 1
                idx = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if starts[mid] <= time:
                        idx = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if idx != -1:
                    p = programs[idx]
                    if p.start <= time < p.end:
                        return p

        # fallback: linear scan of channel programs
        for program in channel.programs:
            if program.start <= time < program.end:
                return program

    @staticmethod
    def get_program_by_unique_id(instance_data: InstanceData | None, unique_id: str) -> Optional[object]:
        # use cache if available, else linear search across all programs
        effective_instance = instance_data if instance_data is not None else Utils._current_instance
        if effective_instance is None:
            return None
        if Utils._unique_id_to_program is not None:
            return Utils._unique_id_to_program.get(unique_id)
        for ch in effective_instance.channels:
            for p in ch.programs:
                if p.unique_id == unique_id:
                    return p


