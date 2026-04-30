namespace SchedulingAPI.Models
{
    public class PriorityBlock
    {
        public int Start { get; set; }
        public int End { get; set; }
        public List<int> AllowedChannels { get; set; } = new();
    }
}

