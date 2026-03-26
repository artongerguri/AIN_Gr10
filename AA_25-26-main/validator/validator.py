from typing import List

from models.instance_data import InstanceData
from models.schedule import Schedule
from utils.utils import Utils
from validator.exceptions.constraint_exception import ConstraintException


class Validator:

    @staticmethod
    def is_channel_valid(schedule_plan: List[Schedule], instance_data: InstanceData, channel_index: int,
                         schedule_time: int):
        try:
            Validator.validate_schedule_time(instance_data, schedule_time)
            Validator.validate_min_duration(schedule_plan, instance_data, schedule_time)
            Validator.validate_max_consecutive_genre(schedule_plan, instance_data, channel_index, schedule_time)
            Validator.validate_priority_time_block(instance_data, channel_index, schedule_time)
        except ConstraintException:
            return False

        return True

    @staticmethod
    def validate_schedule_time(instance_data: InstanceData, schedule_time: int):
        # Check if the schedule time is within bounds and has enough room for min_duration
        if (schedule_time < instance_data.opening_time or 
                schedule_time >= instance_data.closing_time or
                schedule_time + instance_data.min_duration > instance_data.closing_time):
            raise ConstraintException("Schedule time is invalid!")

    @staticmethod
    def validate_min_duration(schedule_plan: List[Schedule], instance_data: InstanceData, schedule_time: int):
        if not schedule_plan:
            return

        last_schedule = schedule_plan[-1]
        if schedule_time < last_schedule.start + instance_data.min_duration:
            raise ConstraintException("min_duration for broadcasting channel has not been reached.")

    @staticmethod
    def validate_max_consecutive_genre(schedule_plan: List[Schedule], instance_data: InstanceData, channel_index: int,
                                       schedule_time: int):
        if not schedule_plan:
            return

        channel_to_insert = instance_data.channels[channel_index]
        program = Utils.get_channel_program_by_time(channel_to_insert, schedule_time)

        if not program:
            return

        count = 0
        for schedule in reversed(schedule_plan):
            scheduled_program = Utils.get_program_by_unique_id(instance_data, schedule.unique_program_id)
            if scheduled_program.genre != program.genre:
                break
            count += 1

        # Max R consecutive means we can have R programs, so reject if count + 1 > R
        if count + 1 > instance_data.max_consecutive_genre:
            raise ConstraintException("max consecutive genre has been reached.")

    @staticmethod
    def validate_priority_time_block(instance_data: InstanceData, channel_index: int, schedule_time: int):
        channel_to_insert = instance_data.channels[channel_index]
        channel_to_insert_id = channel_to_insert.channel_id
        
        # Get the actual program that would be scheduled
        program = Utils.get_channel_program_by_time(channel_to_insert, schedule_time)
        if not program:
            return
        
        # Check if the program's duration overlaps with any priority block
        for block in instance_data.priority_blocks:
            # Check if program's time range [program.start, program.end) overlaps with block [block.start, block.end)
            if (program.start < block.end and program.end > block.start and 
                    channel_to_insert_id not in block.allowed_channels):
                raise ConstraintException("Channel not allowed in priority block.")
