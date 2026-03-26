from typing import List

from models.channel import Channel
from models.instance_data import InstanceData
from models.program import Program
from models.schedule import Schedule
from utils.utils import Utils


class AlgorithmUtils:

    @staticmethod
    def get_best_fit(scheduled_programs: List[Schedule], instance_data: InstanceData, schedule_time: int,
                     valid_channel_indexes: List[int]) ->  tuple[Channel, Program, int]:

        # returns best channel to pick at the time and what score will it provide if we switch to it

        max_score = 0
        best_channel = None
        best_program = None

        for channel_index in valid_channel_indexes:
            channel = instance_data.channels[channel_index]
            program = Utils.get_channel_program_by_time(channel, schedule_time)

            if not program:
                continue

            score = 0

            score += program.score
            score += AlgorithmUtils.get_time_preference_bonus(instance_data, program, schedule_time)
            score += AlgorithmUtils.get_switch_penalty(scheduled_programs, instance_data, channel)
            score += AlgorithmUtils.get_delay_penalty(scheduled_programs, instance_data, program, schedule_time)
            score += AlgorithmUtils.get_early_termination_penalty(scheduled_programs, instance_data, program, schedule_time)

            if score > max_score:
                max_score = score
                best_channel = channel
                best_program = program

        return best_channel, best_program, max_score

    @staticmethod
    def get_time_preference_bonus(instance_data: InstanceData, program: Program, schedule_time: int):
        """
        Calculate time preference bonus for a program.
        CRITICAL: The bonus only applies if the program overlaps with the preferred time window
        for AT LEAST the minimum duration D minutes.

        Per PDF specification: "In order to collect the bonus, the program must fall within
        the preferred interval with at least the minimum scheduled duration D."
        """
        score = 0
        for preference in instance_data.time_preferences:
            if program.genre == preference.preferred_genre:
                # Calculate overlap between program and preference interval
                overlap_start = max(program.start, preference.start)
                overlap_end = min(program.end, preference.end)
                overlap_duration = overlap_end - overlap_start

                # Only award bonus if overlap is at least min_duration
                if overlap_duration >= instance_data.min_duration:
                    score += preference.bonus

        return score

    @staticmethod
    def get_switch_penalty(scheduled_programs: List[Schedule], instance_data: InstanceData, channel: Channel):
        penalty = 0
        if not scheduled_programs:
            return penalty

        last_schedule = scheduled_programs[-1]
        if last_schedule.channel_id != channel.channel_id:
            penalty -= instance_data.switch_penalty

        return penalty

    @staticmethod
    def get_delay_penalty(scheduled_programs: List[Schedule], instance_data: InstanceData, program: Program,
                          schedule_time: int):
        """
        Penalty for switching to a program after its scheduled start time.
        Since we now always schedule programs at their original times, this should not apply.
        """
        penalty = 0
        # No delay penalty - we always schedule programs at their original start time
        return penalty

    @staticmethod
    def get_early_termination_penalty(scheduled_programs: List[Schedule], instance_data: InstanceData, program: Program,
                                      schedule_time: int):
        """
        Penalty for terminating the previous program before its scheduled end time.
        This occurs when we switch to a new program while the previous one is still running.
        Since we now prevent overlaps, this checks if switching to the new program would
        cut off the previous one early.
        """
        penalty = 0
        if not scheduled_programs:
            return penalty

        last_schedule = scheduled_programs[-1]
        
        # If the new program starts before the previous program's natural end
        # (and we're not continuing the same program), we're cutting it short
        if last_schedule.unique_program_id != program.unique_id and program.start < last_schedule.end:
            penalty -= instance_data.termination_penalty

        return penalty
