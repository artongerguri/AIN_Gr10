# ASP.NET Core Scheduling API - Quick Start Guide

## What is this?

This is an ASP.NET Core Web API that implements the TV program scheduling algorithm. It accepts JSON input describing TV channels, programs, and constraints, then returns an optimized schedule.

## Prerequisites

- .NET 9.0 SDK or higher

## Quick Start

### 1. Run the API

Open a terminal in the `SchedulingAPI` directory and run:

```powershell
./run.ps1
```

Or manually:

```powershell
dotnet run
```

The API will start at: `http://localhost:5191`

### 2. View API Documentation

Open your browser and navigate to:

```
http://localhost:5191/swagger
```

You'll see an interactive API documentation page where you can test the endpoint.

### 3. Test the API (Programmatically)

In a **new terminal window**, run:

```powershell
./test-api.ps1
```

This will send the `example_input.json` file to the API and display the results.

## Manual Testing with cURL or Postman

### Using PowerShell (Windows)

```powershell
$json = Get-Content example_input.json -Raw
Invoke-RestMethod -Uri "http://localhost:5191/api/schedule" -Method Post -ContentType "application/json" -Body $json | ConvertTo-Json -Depth 10
```

### Using cURL (Linux/Mac/Windows)

```bash
curl -X POST http://localhost:5191/api/schedule \
  -H "Content-Type: application/json" \
  -d @example_input.json
```

## Understanding the Input/Output

### Input Format

The API accepts JSON with this structure:

```json
{
  "opening_time": 540,      // Start time in minutes from midnight (9:00 AM)
  "closing_time": 1080,     // End time in minutes (6:00 PM)
  "min_duration": 30,       // Minimum program duration in minutes
  "max_consecutive_genre": 2, // Max consecutive programs of same genre
  "channels_count": 3,
  "switch_penalty": 5,      // Penalty for switching channels
  "termination_penalty": 10, // Penalty for early termination
  "priority_blocks": [...], // Time blocks with channel restrictions
  "time_preferences": [...], // Genre preferences with bonuses
  "channels": [...]         // Channel and program data
}
```

### Output Format

The API returns:

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

## Project Structure

```
SchedulingAPI/
├── Controllers/          # API endpoints
├── Services/            # Business logic (schedulers)
├── Models/              # Domain models
├── DTOs/                # Data Transfer Objects
├── Utils/               # Utility functions
├── example_input.json   # Sample input
├── run.ps1             # Start server script
└── test-api.ps1        # Test script
```

## Troubleshooting

### Port Already in Use

If you get a port conflict error, edit `Properties/launchSettings.json` and change the port number.

### Build Errors

Restore packages:

```powershell
dotnet restore
dotnet build
```

### Cannot Connect

Make sure the server is running before testing the API.

## Next Steps

1. Try modifying `example_input.json` with your own data
2. Explore the Swagger UI at `/swagger`
3. Check the README.md for detailed algorithm information
4. Review the source code to understand the implementation

## Related Files

- See `../main.py` for the original Python implementation
- See `../data/input/` for more example input files

