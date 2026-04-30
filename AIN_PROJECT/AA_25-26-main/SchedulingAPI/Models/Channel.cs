namespace SchedulingAPI.Models
{
    public class Channel
    {
        public int ChannelId { get; set; }
        public string ChannelName { get; set; } = string.Empty;
        public List<TVProgram> Programs { get; set; } = new();

        public override string ToString()
        {
            return $"Channel({ChannelId}, {ChannelName}, Programs: {Programs.Count})";
        }
    }
}

