class PriorityBlock:
    def __init__(self, start, end, allowed_channels):
        self.start = start
        self.end = end
        self.allowed_channels = allowed_channels

    def __repr__(self):
        return f"PriorityBlock({self.start}-{self.end}, Allowed: {self.allowed_channels})"