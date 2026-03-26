using SchedulingAPI.Models;

namespace SchedulingAPI.Utils
{
    public static class Validator
    {
        public static bool IsChannelValid(List<Schedule> schedulePlan, InstanceData instanceData, int channelIndex, int scheduleTime)
        {
            try
            {
                ValidateScheduleTime(instanceData, scheduleTime);
                ValidateMinDuration(schedulePlan, instanceData, scheduleTime);
                ValidateMaxConsecutiveGenre(schedulePlan, instanceData, channelIndex, scheduleTime);
                ValidatePriorityTimeBlock(instanceData, channelIndex, scheduleTime);
            }
            catch (Exception)
            {
                return false;
            }

            return true;
        }

        private static void ValidateScheduleTime(InstanceData instanceData, int scheduleTime)
        {
            if (scheduleTime < instanceData.OpeningTime ||
                scheduleTime >= instanceData.ClosingTime ||
                scheduleTime + instanceData.MinDuration > instanceData.ClosingTime)
            {
                throw new Exception("Schedule time is invalid!");
            }
        }

        private static void ValidateMinDuration(List<Schedule> schedulePlan, InstanceData instanceData, int scheduleTime)
        {
            if (schedulePlan.Count == 0) return;

            var lastSchedule = schedulePlan[^1];
            if (scheduleTime < lastSchedule.Start + instanceData.MinDuration)
            {
                throw new Exception("min_duration for broadcasting channel has not been reached.");
            }
        }

        private static void ValidateMaxConsecutiveGenre(List<Schedule> schedulePlan, InstanceData instanceData, int channelIndex, int scheduleTime)
        {
            if (schedulePlan.Count == 0) return;

            var channelToInsert = instanceData.Channels[channelIndex];
            var program = Utils.GetChannelProgramByTime(channelToInsert, scheduleTime);

            if (program == null) return;

            int count = 0;
            for (int i = schedulePlan.Count - 1; i >= 0; i--)
            {
                var schedule = schedulePlan[i];
                var scheduledProgram = Utils.GetProgramByUniqueId(instanceData, schedule.UniqueProgramId);
                if (scheduledProgram == null || scheduledProgram.Genre != program.Genre)
                    break;
                count++;
            }

            if (count + 1 > instanceData.MaxConsecutiveGenre)
            {
                throw new Exception("max consecutive genre has been reached.");
            }
        }

        private static void ValidatePriorityTimeBlock(InstanceData instanceData, int channelIndex, int scheduleTime)
        {
            var channelToInsert = instanceData.Channels[channelIndex];
            var channelToInsertId = channelToInsert.ChannelId;

            var program = Utils.GetChannelProgramByTime(channelToInsert, scheduleTime);
            if (program == null) return;

            foreach (var block in instanceData.PriorityBlocks)
            {
                if (program.Start < block.End && program.End > block.Start &&
                    !block.AllowedChannels.Contains(channelToInsertId))
                {
                    throw new Exception("Channel not allowed in priority block.");
                }
            }
        }
    }
}

