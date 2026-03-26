using Microsoft.AspNetCore.Mvc;
using SchedulingAPI.DTOs;
using SchedulingAPI.Models;
using SchedulingAPI.Services;
using SchedulingAPI.Utils;

namespace SchedulingAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ScheduleController : ControllerBase
    {
        private readonly ILogger<ScheduleController> _logger;

        public ScheduleController(ILogger<ScheduleController> logger)
        {
            _logger = logger;
        }

        [HttpPost]
        public ActionResult<ScheduleResponse> GenerateSchedule([FromBody] ScheduleRequest request)
        {

            try
            {
                var instanceData = MapToInstanceData(request);
                
                Utils.Utils.SetCurrentInstance(instanceData);

                var scheduler = new GreedyScheduler(instanceData);
                var solution = scheduler.GenerateSolution();

                _logger.LogInformation("Generated solution with total score: {TotalScore}", solution.TotalScore);

                // Convert solution to response DTO
                var response = new ScheduleResponse
                {
                    TotalScore = solution.TotalScore,
                    ScheduledPrograms = solution.ScheduledPrograms.Select(s => new ScheduledProgramDto
                    {
                        ProgramId = s.ProgramId,
                        ChannelId = s.ChannelId,
                        Start = s.Start,
                        End = s.End
                    }).ToList()
                };

                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating schedule");
                return StatusCode(500, new { error = "An error occurred while generating the schedule", details = ex.Message });
            }
        }

        private InstanceData MapToInstanceData(ScheduleRequest request)
        {
            var channels = request.Channels.Select(c => new Channel
            {
                ChannelId = c.ChannelId,
                ChannelName = c.ChannelName ?? $"Channel {c.ChannelId}",
                Programs = c.Programs.Select(p => new TVProgram
                {
                    ProgramId = p.ProgramId,
                    Start = p.Start,
                    End = p.End,
                    Genre = p.Genre,
                    Score = p.Score,
                    UniqueId = $"{c.ChannelId}_{p.ProgramId}_{p.Start}"
                }).ToList()
            }).ToList();

            return new InstanceData
            {
                OpeningTime = request.OpeningTime,
                ClosingTime = request.ClosingTime,
                MinDuration = request.MinDuration,
                MaxConsecutiveGenre = request.MaxConsecutiveGenre,
                ChannelsCount = request.ChannelsCount,
                SwitchPenalty = request.SwitchPenalty,
                TerminationPenalty = request.TerminationPenalty,
                PriorityBlocks = request.PriorityBlocks.Select(pb => new PriorityBlock
                {
                    Start = pb.Start,
                    End = pb.End,
                    AllowedChannels = pb.AllowedChannels
                }).ToList(),
                TimePreferences = request.TimePreferences.Select(tp => new TimePreference
                {
                    Start = tp.Start,
                    End = tp.End,
                    PreferredGenre = tp.PreferredGenre,
                    Bonus = tp.Bonus
                }).ToList(),
                Channels = channels
            };
        }
    }
}

