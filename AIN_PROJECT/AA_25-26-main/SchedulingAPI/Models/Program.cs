namespace SchedulingAPI.Models
{
    public class TVProgram
    {
        public string ProgramId { get; set; } = string.Empty;
        public int Start { get; set; }
        public int End { get; set; }
        public string Genre { get; set; } = string.Empty;
        public int Score { get; set; }
        public string UniqueId { get; set; } = string.Empty;

        public override string ToString()
        {
            return $"TVProgram(ID:{UniqueId}, {ProgramId}, {Start}-{End}, {Genre}, {Score})";
        }
    }
}

