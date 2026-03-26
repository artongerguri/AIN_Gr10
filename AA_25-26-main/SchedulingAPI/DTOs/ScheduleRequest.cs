using System.Text.Json.Serialization;

namespace SchedulingAPI.DTOs
{
    public class ScheduleRequest
    {
        [JsonPropertyName("opening_time")]
        public int OpeningTime { get; set; }

        [JsonPropertyName("closing_time")]
        public int ClosingTime { get; set; }

        [JsonPropertyName("min_duration")]
        public int MinDuration { get; set; }

        [JsonPropertyName("max_consecutive_genre")]
        public int MaxConsecutiveGenre { get; set; }

        [JsonPropertyName("channels_count")]
        public int ChannelsCount { get; set; }

        [JsonPropertyName("switch_penalty")]
        public int SwitchPenalty { get; set; }

        [JsonPropertyName("termination_penalty")]
        public int TerminationPenalty { get; set; }

        [JsonPropertyName("priority_blocks")]
        public List<PriorityBlockDto> PriorityBlocks { get; set; } = new();

        [JsonPropertyName("time_preferences")]
        public List<TimePreferenceDto> TimePreferences { get; set; } = new();

        [JsonPropertyName("channels")]
        public List<ChannelDto> Channels { get; set; } = new();
    }

    public class PriorityBlockDto
    {
        [JsonPropertyName("start")]
        public int Start { get; set; }

        [JsonPropertyName("end")]
        public int End { get; set; }

        [JsonPropertyName("allowed_channels")]
        public List<int> AllowedChannels { get; set; } = new();
    }

    public class TimePreferenceDto
    {
        [JsonPropertyName("start")]
        public int Start { get; set; }

        [JsonPropertyName("end")]
        public int End { get; set; }

        [JsonPropertyName("preferred_genre")]
        public string PreferredGenre { get; set; } = string.Empty;

        [JsonPropertyName("bonus")]
        public int Bonus { get; set; }
    }

    public class ChannelDto
    {
        [JsonPropertyName("channel_id")]
        public int ChannelId { get; set; }

        [JsonPropertyName("channel_name")]
        public string? ChannelName { get; set; }

        [JsonPropertyName("programs")]
        public List<ProgramDto> Programs { get; set; } = new();
    }

    public class ProgramDto
    {
        [JsonPropertyName("program_id")]
        public string ProgramId { get; set; } = string.Empty;

        [JsonPropertyName("start")]
        public int Start { get; set; }

        [JsonPropertyName("end")]
        public int End { get; set; }

        [JsonPropertyName("genre")]
        public string Genre { get; set; } = string.Empty;

        [JsonPropertyName("score")]
        public int Score { get; set; }
    }
}

