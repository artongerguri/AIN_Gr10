using SchedulingAPI.Models;
using SchedulingAPI.Utils;

namespace SchedulingAPI.Services
{
    public class GreedyScheduler
    {
        private readonly InstanceData _instanceData;

        public GreedyScheduler(InstanceData instanceData)
        {
            _instanceData = instanceData;
        }

        public Solution GenerateSolution()
        {
            int time = _instanceData.OpeningTime;
            int totalScore = 0;
            var solution = new List<Schedule>();

            while (time < _instanceData.ClosingTime)
            {
                var validChannelIndexes = SchedulerUtils.GetValidSchedules(solution, _instanceData, time);
                
                if (validChannelIndexes.Count == 0)
                {
                    time++;
                    continue;
                }

                var (bestChannel, channelProgram, fitness) = AlgorithmUtils.GetBestFit(
                    solution, _instanceData, time, validChannelIndexes);

                if (bestChannel == null || channelProgram == null || fitness <= 0)
                {
                    time++;
                    continue;
                }

                // Check if we're trying to schedule the exact same program again
                if (solution.Count > 0 && solution[^1].UniqueProgramId == channelProgram.UniqueId)
                {
                    time++;
                    continue;
                }

                // Check if this program would overlap with the previous program
                if (solution.Count > 0 && channelProgram.Start < solution[^1].End)
                {
                    time++;
                    continue;
                }

                // Check if this program meets minimum duration requirement
                if (channelProgram.End - channelProgram.Start < _instanceData.MinDuration)
                {
                    time++;
                    continue;
                }

                var schedule = new Schedule
                {
                    ProgramId = channelProgram.ProgramId,
                    ChannelId = bestChannel.ChannelId,
                    Start = channelProgram.Start,
                    End = channelProgram.End,
                    Fitness = fitness,
                    UniqueProgramId = channelProgram.UniqueId
                };

                solution.Add(schedule);
                time = channelProgram.End;
                totalScore += fitness;
            }

            return new Solution
            {
                ScheduledPrograms = solution,
                TotalScore = totalScore
            };
        }
    }
}

