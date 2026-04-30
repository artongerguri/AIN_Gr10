# TV Schedule Optimization API

ASP.NET Core Web API implementation of the TV program scheduling algorithm from the Advanced Algorithms course (2025/26).

## Overview

This API accepts TV channel programming data as JSON input and returns an optimized schedule that maximizes viewer satisfaction while adhering to various constraints.

## Features

- **Greedy Scheduling Algorithm**: Efficient program selection based on scores and constraints
- **Multiple Constraints Support**:
  - Minimum program duration
  - Maximum consecutive programs of the same genre
  - Priority time blocks (restricted channel access during specific times)
  - Time preferences with bonus scores
  - Channel switching penalties
  - Early program termination penalties

## Project Structure

```
SchedulingAPI/
├── Controllers/
│   └── ScheduleController.cs      # API endpoint
├── Models/
│   ├── Channel.cs                 # Channel model
│   ├── Program.cs                 # Program model
│   ├── InstanceData.cs            # Main input data model
│   ├── Schedule.cs                # Scheduled program model
│   ├── Solution.cs                # Solution model
│   ├── PriorityBlock.cs           # Priority block constraint
│   └── TimePreference.cs          # Time preference bonus
├── DTOs/
│   ├── ScheduleRequest.cs         # Input DTO
│   └── ScheduleResponse.cs        # Output DTO
├── Services/
│   └── GreedyScheduler.cs         # Scheduling algorithm
├── Utils/
│   ├── Utils.cs                   # Utility functions
│   ├── Validator.cs               # Constraint validation
│   ├── SchedulerUtils.cs          # Scheduling helpers
│   └── AlgorithmUtils.cs          # Algorithm calculations
└── Program.cs                      # Application entry point
```

## Getting Started

### Prerequisites

- .NET 9.0 SDK or higher
- Any IDE (Visual Studio, VS Code, Rider, etc.)

### Running the API

1. Navigate to the SchedulingAPI directory:
```bash
cd SchedulingAPI
```

2. Restore dependencies:
```bash
dotnet restore
```

3. Run the application:
```bash
dotnet run
```

The API will start on `http://localhost:5000` (or the port specified in launchSettings.json).

### Using Swagger UI

Navigate to `http://localhost:5000/swagger` in your browser to access the interactive API documentation.

## API Endpoint

### POST /api/schedule

Generates an optimized TV schedule based on input constraints.

**Request Body:**

```json
{
  "opening_time": 540,
  "closing_time": 1080,
  "min_duration": 30,
  "max_consecutive_genre": 2,
  "channels_count": 3,
  "switch_penalty": 5,
  "termination_penalty": 10,
  "priority_blocks": [
    {
      "start": 720,
      "end": 780,
      "allowed_channels": [0, 2]
    }
  ],
  "time_preferences": [
    {
      "start": 540,
      "end": 720,
      "preferred_genre": "news",
      "bonus": 50
    }
  ],
  "channels": [
    {
      "channel_id": 0,
      "channel_name": "Channel 1",
      "programs": [
        {
          "program_id": "n1",
          "start": 540,
          "end": 600,
          "genre": "news",
          "score": 80
        }
      ]
    }
  ]
}
```

**Response:**

```json
{
  "scheduled_programs": [
    {
      "program_id": "n1",
      "channel_id": 0,
      "start": 540,
      "end": 600
    }
  ],
  "total_score": 130
}
```

## Input Parameters

- **opening_time**: Start time of the broadcast day (in minutes from midnight)
- **closing_time**: End time of the broadcast day (in minutes from midnight)
- **min_duration**: Minimum duration for each scheduled program (in minutes)
- **max_consecutive_genre**: Maximum number of consecutive programs of the same genre
- **channels_count**: Total number of available channels
- **switch_penalty**: Score penalty for switching between channels
- **termination_penalty**: Score penalty for terminating a program before its natural end
- **priority_blocks**: Time windows with channel restrictions
- **time_preferences**: Time windows with genre preferences and bonus scores
- **channels**: Array of channels with their programs

## Testing with cURL

```bash
curl -X POST http://localhost:5000/api/schedule \
  -H "Content-Type: application/json" \
  -d @path/to/input.json
```

## Algorithm Details

The Greedy Scheduler works by:

1. Starting at the opening time
2. Finding all valid channels at the current time (based on constraints)
3. Calculating fitness scores for each valid option considering:
   - Program base score
   - Time preference bonuses
   - Channel switch penalties
   - Early termination penalties
4. Selecting the program with the highest fitness
5. Moving time forward to the end of the selected program
6. Repeating until closing time

## Related Projects

This is a C# port of the Python implementation in the parent directory, part of the Advanced Algorithms course at the University of Prishtina.

## License

See LICENSE file in the root directory.

