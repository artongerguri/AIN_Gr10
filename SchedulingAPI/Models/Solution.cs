namespace SchedulingAPI.Models
{
    public class Solution
    {
        public List<Schedule> ScheduledPrograms { get; set; } = new();
        public int TotalScore { get; set; }

        public override string ToString()
        {
            return $"Solution(TotalScore={TotalScore}, ScheduledPrograms Count={ScheduledPrograms.Count})";
        }
    }
}

