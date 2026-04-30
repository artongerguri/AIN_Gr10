using SchedulingAPI.Models;

namespace SchedulingAPI.Utils
{
    public static class AlgorithmUtils
    {
        public static (Channel? bestChannel, TVProgram? bestProgram, int maxScore) GetBestFit(
            List<Schedule> scheduledPrograms, 
            InstanceData instanceData, 
            int scheduleTime,
            List<int> validChannelIndexes)
        {
            int maxScore = 0;
            Channel? bestChannel = null;
            TVProgram? bestProgram = null;

            foreach (var channelIndex in validChannelIndexes)
            {
                var channel = instanceData.Channels[channelIndex];
                var program = Utils.GetChannelProgramByTime(channel, scheduleTime);

                if (program == null) continue;

                int score = program.Score;
                score += GetTimePreferenceBonus(instanceData, program, scheduleTime);
                score += GetSwitchPenalty(scheduledPrograms, instanceData, channel);
                score += GetDelayPenalty(scheduledPrograms, instanceData, program, scheduleTime);
                score += GetEarlyTerminationPenalty(scheduledPrograms, instanceData, program, scheduleTime);

                if (score > maxScore)
                {
                    maxScore = score;
                    bestChannel = channel;
                    bestProgram = program;
                }
            }

            return (bestChannel, bestProgram, maxScore);
        }

        public static int GetTimePreferenceBonus(InstanceData instanceData, TVProgram program, int scheduleTime)
        {
            int score = 0;
            foreach (var preference in instanceData.TimePreferences)
            {
                if (program.Genre == preference.PreferredGenre)
                {
                    if (program.Start < preference.End && program.End > preference.Start)
                    {
                        score += preference.Bonus;
                    }
                }
            }
            return score;
        }

        public static int GetSwitchPenalty(List<Schedule> scheduledPrograms, InstanceData instanceData, Channel channel)
        {
            int penalty = 0;
            if (scheduledPrograms.Count == 0) return penalty;

            var lastSchedule = scheduledPrograms[^1];
            if (lastSchedule.ChannelId != channel.ChannelId)
            {
                penalty -= instanceData.SwitchPenalty;
            }

            return penalty;
        }

        public static int GetDelayPenalty(List<Schedule> scheduledPrograms, InstanceData instanceData, TVProgram program, int scheduleTime)
        {
            // No delay penalty - we always schedule programs at their original start time
            return 0;
        }

        public static int GetEarlyTerminationPenalty(List<Schedule> scheduledPrograms, InstanceData instanceData, TVProgram program, int scheduleTime)
        {
            int penalty = 0;
            if (scheduledPrograms.Count == 0) return penalty;

            var lastSchedule = scheduledPrograms[^1];

            if (lastSchedule.UniqueProgramId != program.UniqueId && program.Start < lastSchedule.End)
            {
                penalty -= instanceData.TerminationPenalty;
            }

            return penalty;
        }
    }
}

