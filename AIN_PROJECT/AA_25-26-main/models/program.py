class Program:
    def __init__(self, program_id, start, end, genre, score, unique_id=None):
        self.program_id = program_id
        self.start = start
        self.end = end
        self.genre = genre
        self.score = score
        self.unique_id = unique_id

    def __repr__(self):
        return f"Program(ID:{self.unique_id}, {self.program_id}, {self.start}-{self.end}, {self.genre}, {self.score})"
