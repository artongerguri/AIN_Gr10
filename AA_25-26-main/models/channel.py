class Channel:
    def __init__(self, channel_id, channel_name, programs):
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.programs = programs

    def __repr__(self):
        return f"Channel({self.channel_id}, {self.channel_name}, Programs: {len(self.programs)})"
