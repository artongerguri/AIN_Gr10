namespace SchedulingAPI.Models
{
    public class TimePreference
    {
        public int Start { get; set; }
        public int End { get; set; }
        public string PreferredGenre { get; set; } = string.Empty;
        public int Bonus { get; set; }
    }
}

