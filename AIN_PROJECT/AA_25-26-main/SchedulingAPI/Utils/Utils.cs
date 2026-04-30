using SchedulingAPI.Models;

namespace SchedulingAPI.Utils
{
    public static class Utils
    {
        private static InstanceData? _currentInstance;
        private static Dictionary<int, List<TVProgram>>? _channelToSortedPrograms;
        private static Dictionary<int, List<int>>? _channelToStarts;
        private static Dictionary<string, TVProgram>? _uniqueIdToProgram;

        public static void SetCurrentInstance(InstanceData instanceData)
        {
            _currentInstance = instanceData;
            BuildCaches();
        }

        private static void BuildCaches()
        {
            if (_currentInstance == null) return;

            _channelToSortedPrograms = new Dictionary<int, List<TVProgram>>();
            _channelToStarts = new Dictionary<int, List<int>>();
            _uniqueIdToProgram = new Dictionary<string, TVProgram>();

            foreach (var channel in _currentInstance.Channels)
            {
                var sortedPrograms = channel.Programs.OrderBy(p => p.Start).ToList();
                _channelToSortedPrograms[channel.ChannelId] = sortedPrograms;
                _channelToStarts[channel.ChannelId] = sortedPrograms.Select(p => p.Start).ToList();

                foreach (var program in sortedPrograms)
                {
                    if (!string.IsNullOrEmpty(program.UniqueId))
                    {
                        _uniqueIdToProgram[program.UniqueId] = program;
                    }
                }
            }
        }

        public static TVProgram? GetChannelProgramByTime(Channel channel, int time)
        {
            if (_currentInstance != null && _channelToSortedPrograms != null && _channelToStarts != null)
            {
                if (_channelToSortedPrograms.TryGetValue(channel.ChannelId, out var programs) &&
                    _channelToStarts.TryGetValue(channel.ChannelId, out var starts))
                {
                    // Binary search for rightmost start <= time
                    int lo = 0, hi = starts.Count - 1;
                    int idx = -1;
                    
                    while (lo <= hi)
                    {
                        int mid = (lo + hi) / 2;
                        if (starts[mid] <= time)
                        {
                            idx = mid;
                            lo = mid + 1;
                        }
                        else
                        {
                            hi = mid - 1;
                        }
                    }

                    if (idx != -1)
                    {
                        var program = programs[idx];
                        if (program.Start <= time && time < program.End)
                        {
                            return program;
                        }
                    }
                }
            }

            // Fallback: linear scan
            return channel.Programs.FirstOrDefault(p => p.Start <= time && time < p.End);
        }

        public static TVProgram? GetProgramByUniqueId(InstanceData? instanceData, string uniqueId)
        {
            var effectiveInstance = instanceData ?? _currentInstance;
            if (effectiveInstance == null) return null;

            if (_uniqueIdToProgram != null && _uniqueIdToProgram.TryGetValue(uniqueId, out var cachedProgram))
            {
                return cachedProgram;
            }

            // Fallback: linear search
            foreach (var channel in effectiveInstance.Channels)
            {
                var program = channel.Programs.FirstOrDefault(p => p.UniqueId == uniqueId);
                if (program != null) return program;
            }

            return null;
        }
    }
}

