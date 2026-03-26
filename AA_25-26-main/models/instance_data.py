class InstanceData:
    def __init__(self, opening_time, closing_time, min_duration, max_consecutive_genre,
                 channels_count, switch_penalty, termination_penalty,
                 priority_blocks, time_preferences, channels):
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.min_duration = min_duration
        self.max_consecutive_genre = max_consecutive_genre
        self.channels_count = channels_count
        self.switch_penalty = switch_penalty
        self.termination_penalty = termination_penalty
        self.priority_blocks = priority_blocks
        self.time_preferences = time_preferences
        self.channels = channels

    def __repr__(self):
        return (f"InstanceData(\n"
                f"  opening_time={self.opening_time},\n"
                f"  closing_time={self.closing_time},\n"
                f"  min_duration={self.min_duration},\n"
                f"  max_consecutive_genre={self.max_consecutive_genre},\n"
                f"  channels_count={self.channels_count},\n"
                f"  switch_penalty={self.switch_penalty},\n"
                f"  termination_penalty={self.termination_penalty},\n"
                f"  priority_blocks={self.priority_blocks},\n"
                f"  time_preferences={self.time_preferences},\n"
                f"  channels={self.channels}\n"
                f")")
