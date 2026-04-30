class TimePreference:
    def __init__(self, start, end, preferred_genre, bonus):
        self.start = start
        self.end = end
        self.preferred_genre = preferred_genre
        self.bonus = bonus

    def __repr__(self):
        return f"TimePreference({self.start}-{self.end}, Genre: {self.preferred_genre}, Bonus: {self.bonus})"