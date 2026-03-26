namespace SchedulingAPI.Models
{
    public class InstanceData
    {
        public int OpeningTime { get; set; }
        public int ClosingTime { get; set; }
        public int MinDuration { get; set; }
        public int MaxConsecutiveGenre { get; set; }
        public int ChannelsCount { get; set; }
        public int SwitchPenalty { get; set; }
        public int TerminationPenalty { get; set; }
        public List<PriorityBlock> PriorityBlocks { get; set; } = new();
        public List<TimePreference> TimePreferences { get; set; } = new();
        public List<Channel> Channels { get; set; } = new();
    }
}

