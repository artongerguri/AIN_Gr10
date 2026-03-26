from typing import List

from models.schedule import Schedule


class Solution:
    def __init__(self, scheduled_programs: List[Schedule], total_score: int):
        self.scheduled_programs = scheduled_programs
        self.total_score = total_score

    def __repr__(self):
        return f"Solution({self.total_score}, scheduled_programs: {self.scheduled_programs})"