using System.Text.Json.Serialization;

namespace SchedulingAPI.DTOs
{
    public class ScheduleResponse
    {
        [JsonPropertyName("scheduled_programs")]
        public List<ScheduledProgramDto> ScheduledPrograms { get; set; } = new();

        [JsonPropertyName("total_score")]
        public int TotalScore { get; set; }
    }

    public class ScheduledProgramDto
    {
        [JsonPropertyName("program_id")]
        public string ProgramId { get; set; } = string.Empty;

        [JsonPropertyName("channel_id")]
        public int ChannelId { get; set; }

        [JsonPropertyName("start")]
        public int Start { get; set; }

        [JsonPropertyName("end")]
        public int End { get; set; }
    }
}

