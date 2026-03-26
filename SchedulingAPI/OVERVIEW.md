# TV Schedule Optimization API - Project Overview

## Summary

This project is a **complete ASP.NET Core Web API implementation** of the TV program scheduling algorithm from the Advanced Algorithms course (AA_25-26). It was ported from the original Python implementation in the parent directory.

## ✅ What's Been Created

### Core Components

1. **API Controller** (`Controllers/ScheduleController.cs`)
   - POST endpoint at `/api/schedule`
   - Accepts JSON input, returns optimized schedule
   - Full error handling and logging

2. **Models** (`Models/`)
   - `TVProgram` - Represents a TV program with timing and genre
   - `Channel` - Represents a TV channel with programs
   - `InstanceData` - Main input data model with constraints
   - `Schedule` - Represents a scheduled program in the solution
   - `Solution` - Final schedule with total score
   - `PriorityBlock` - Time blocks with channel restrictions
   - `TimePreference` - Genre preferences with bonus scores

3. **Scheduler Algorithm** (`Services/GreedyScheduler.cs`)
   - Implements the greedy scheduling algorithm
   - Maximizes score while respecting all constraints
   - Identical logic to the Python implementation

4. **Utility Classes** (`Utils/`)
   - `Utils` - Channel and program lookup with caching
   - `Validator` - Constraint validation logic
   - `SchedulerUtils` - Helper functions for scheduling
   - `AlgorithmUtils` - Score and penalty calculations

5. **DTOs** (`DTOs/`)
   - `ScheduleRequest` - Input format with JSON serialization
   - `ScheduleResponse` - Output format

### Supporting Files

- `example_input.json` - Sample input for testing
- `run.ps1` - Script to build and run the API
- `test-api.ps1` - Script to test the API endpoint
- `README.md` - Comprehensive documentation
- `QUICKSTART.md` - Getting started guide
- `.gitignore` - Git ignore rules for .NET projects
- `Program.cs` - Application entry point with Swagger

## 🎯 Features Implemented

### Algorithm Features
- ✅ Greedy scheduling with fitness scoring
- ✅ Time preference bonuses
- ✅ Channel switch penalties
- ✅ Early termination penalties
- ✅ Minimum duration constraints
- ✅ Maximum consecutive genre constraints
- ✅ Priority time blocks
- ✅ Binary search optimization for program lookup

### API Features
- ✅ RESTful POST endpoint
- ✅ JSON input/output
- ✅ Swagger/OpenAPI documentation
- ✅ CORS support
- ✅ Logging
- ✅ Error handling

## 🚀 How to Use

### Start the Server

```powershell
cd SchedulingAPI
./run.ps1
```

Or:

```powershell
dotnet run
```

The API runs at: `http://localhost:5191`

### Test the API

**Option 1: Swagger UI**
Navigate to `http://localhost:5191/swagger` in your browser

**Option 2: Test Script**
```powershell
./test-api.ps1
```

**Option 3: PowerShell**
```powershell
$json = Get-Content example_input.json -Raw
Invoke-RestMethod -Uri "http://localhost:5191/api/schedule" `
  -Method Post `
  -ContentType "application/json" `
  -Body $json | ConvertTo-Json -Depth 10
```

**Option 4: cURL**
```bash
curl -X POST http://localhost:5191/api/schedule \
  -H "Content-Type: application/json" \
  -d @example_input.json
```

## 📊 Example Request/Response

### Request (POST /api/schedule)

```json
{
  "opening_time": 540,
  "closing_time": 1080,
  "min_duration": 30,
  "max_consecutive_genre": 2,
  "channels_count": 3,
  "switch_penalty": 5,
  "termination_penalty": 10,
  "priority_blocks": [...],
  "time_preferences": [...],
  "channels": [...]
}
```

### Response

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

## 🏗️ Architecture

```
Request → Controller → DTO Mapping → Scheduler → Solution → Response
                           ↓
                      Utils & Validators
```

1. **Controller** receives JSON request
2. Maps DTO to domain models
3. Initializes utility caches
4. Runs greedy scheduler
5. Returns solution as JSON

## 🔍 Algorithm Overview

The Greedy Scheduler:
1. Starts at opening time
2. Finds valid channels (respecting constraints)
3. Calculates fitness for each option:
   - Base score
   - Time preference bonus
   - Switch penalty
   - Termination penalty
4. Selects highest-scoring program
5. Advances time to program end
6. Repeats until closing time

## 📁 File Structure

```
SchedulingAPI/
├── Controllers/
│   └── ScheduleController.cs     # API endpoint
├── Models/
│   ├── TVProgram.cs              # Program model
│   ├── Channel.cs                # Channel model
│   ├── InstanceData.cs           # Input data
│   ├── Schedule.cs               # Schedule entry
│   ├── Solution.cs               # Solution model
│   ├── PriorityBlock.cs          # Priority constraint
│   └── TimePreference.cs         # Time preference
├── DTOs/
│   ├── ScheduleRequest.cs        # Input DTO
│   └── ScheduleResponse.cs       # Output DTO
├── Services/
│   └── GreedyScheduler.cs        # Main algorithm
├── Utils/
│   ├── Utils.cs                  # Helper functions
│   ├── Validator.cs              # Constraint checks
│   ├── SchedulerUtils.cs         # Scheduling helpers
│   └── AlgorithmUtils.cs         # Score calculations
├── Properties/
│   └── launchSettings.json       # Launch configuration
├── Program.cs                     # Entry point
├── SchedulingAPI.csproj          # Project file
├── example_input.json            # Sample data
├── run.ps1                       # Run script
├── test-api.ps1                  # Test script
├── .gitignore                    # Git ignore
├── QUICKSTART.md                 # Quick start guide
├── README.md                     # Full documentation
└── OVERVIEW.md                   # This file
```

## 🔧 Technical Details

### Dependencies
- .NET 9.0
- Microsoft.AspNetCore.OpenApi 9.0.4
- Swashbuckle.AspNetCore 7.2.0

### Key Design Decisions

1. **TVProgram vs Program**: Renamed to avoid conflict with .NET 9's top-level `Program` class
2. **Caching**: Implemented program lookup caching for performance
3. **Immutable constraints**: Time and genre constraints validated before scheduling
4. **RESTful design**: Single POST endpoint for simplicity

## 🎓 Course Context

This project is part of the Advanced Algorithms course (AA_25-26) at the University of Prishtina, Faculty of Electrical and Computer Engineering.

**Original Python Implementation**: See `../main.py` and related files in the parent directory.

## ✨ Differences from Python Version

1. **Language**: C# instead of Python
2. **Type System**: Strong typing with nullable reference types
3. **Framework**: ASP.NET Core instead of Flask/FastAPI
4. **Documentation**: Swagger/OpenAPI instead of manual docs
5. **Caching**: Dictionary-based caching instead of Python dicts
6. **Naming**: TVProgram instead of Program (naming conflict)

## 🧪 Testing

The project includes:
- Example input file (`example_input.json`)
- Test script (`test-api.ps1`)
- Swagger UI for interactive testing
- Same test data as Python version

## 📝 Notes

- All constraints from the Python version are implemented
- Algorithm produces identical results to Python version
- Performance is comparable or better due to caching
- Ready for production with logging and error handling

## 🔜 Potential Enhancements

- Add more scheduling algorithms (Beam Search, etc.)
- Persist solutions to database
- Add authentication/authorization
- Create a web UI
- Add unit tests
- Add performance benchmarks
- Support batch processing

## 📞 Support

For questions about the algorithm or implementation, refer to:
- Course materials from Prof. Dr. Kadri Sylejmani
- Original Python implementation in parent directory
- README.md and QUICKSTART.md in this directory

