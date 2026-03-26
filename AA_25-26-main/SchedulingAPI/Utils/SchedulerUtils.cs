using SchedulingAPI.Models;

namespace SchedulingAPI.Utils
{
    public static class SchedulerUtils
    {
        public static List<int> GetValidSchedules(List<Schedule> scheduledPrograms, InstanceData instanceData, int scheduleTime)
        {
            var validChannels = new List<int>();

            for (int channelIndex = 0; channelIndex < instanceData.Channels.Count; channelIndex++)
            {
                if (Validator.IsChannelValid(scheduledPrograms, instanceData, channelIndex, scheduleTime))
                {
                    validChannels.Add(channelIndex);
                }
            }

            return validChannels;
        }
    }
}

