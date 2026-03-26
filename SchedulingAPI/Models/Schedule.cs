namespace SchedulingAPI.Models
{
    public class Schedule
    {
        public string ProgramId { get; set; } = string.Empty;
        public int ChannelId { get; set; }
        public int Start { get; set; }
        public int End { get; set; }
        public int Fitness { get; set; }
        public string UniqueProgramId { get; set; } = string.Empty;

        public override string ToString()
        {
            return $"Schedule(ProgramId={ProgramId}, ChannelId={ChannelId}, Start={Start}, End={End}, Fitness={Fitness}, UniqueProgramId={UniqueProgramId})";
        }
    }
}

