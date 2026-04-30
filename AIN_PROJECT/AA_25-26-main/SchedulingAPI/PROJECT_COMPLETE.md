# ✅ ASP.NET Core Web API - Project Complete!

## 🎉 What Has Been Created

I've successfully created a complete **ASP.NET Core Web API** project that implements your TV scheduling algorithm. The API accepts JSON input and returns optimized schedules.

## 📦 Project Location

```
AA_25-26/
└── SchedulingAPI/          ← Your new Web API project
    ├── Controllers/
    ├── Models/
    ├── Services/
    ├── Utils/
    ├── DTOs/
    └── ... (documentation and scripts)
```

## 🚀 Quick Start (Choose One)

### Option 1: Using the Run Script (Recommended)

```powershell
cd SchedulingAPI
./run.ps1
```

### Option 2: Manual Start

```powershell
cd SchedulingAPI
dotnet run
```

The API will start at: **http://localhost:5191**

## 📖 Interactive Documentation

Once the server is running, open your browser to:

```
http://localhost:5191/swagger
```

You'll see an interactive API interface where you can:
- View the API schema
- Test the endpoint with sample data
- See request/response examples

## 🧪 Testing the API

### Method 1: Test Script (Easiest)

In a **new terminal**:

```powershell
cd SchedulingAPI
./test-api.ps1
```

This sends the example data and displays the results.

### Method 2: PowerShell

```powershell
$json = Get-Content SchedulingAPI/example_input.json -Raw
Invoke-RestMethod -Uri "http://localhost:5191/api/schedule" `
  -Method Post `
  -ContentType "application/json" `
  -Body $json
```

### Method 3: cURL (Any platform)

```bash
curl -X POST http://localhost:5191/api/schedule \
  -H "Content-Type: application/json" \
  -d @SchedulingAPI/example_input.json
```

## 📋 API Endpoint

**URL**: `POST http://localhost:5191/api/schedule`

**Request Body**: JSON with your scheduling data (same format as `toy.json` in `data/input/`)

**Response**: JSON with scheduled programs and total score

## 📚 Documentation Files

I've created several documentation files for you:

1. **QUICKSTART.md** - Fast getting-started guide
2. **README.md** - Complete project documentation
3. **OVERVIEW.md** - Technical overview and architecture
4. **example_input.json** - Sample input for testing

## 🏗️ What's Included

✅ Complete ASP.NET Core Web API project
✅ Greedy Scheduler algorithm (ported from your Python code)
✅ All constraint validations
✅ Swagger/OpenAPI documentation
✅ Example test data
✅ PowerShell scripts for running and testing
✅ Comprehensive documentation
✅ .gitignore for .NET projects
✅ Error handling and logging

## 🎯 Project Structure

```
SchedulingAPI/
├── Controllers/
│   └── ScheduleController.cs       # API endpoint
├── Models/
│   ├── TVProgram.cs                # TV program model
│   ├── Channel.cs                  # Channel model
│   ├── InstanceData.cs             # Input model
│   ├── Schedule.cs                 # Schedule entry
│   ├── Solution.cs                 # Solution model
│   └── ...                         # Other models
├── Services/
│   └── GreedyScheduler.cs          # Your algorithm!
├── Utils/
│   ├── AlgorithmUtils.cs           # Score calculations
│   ├── Validator.cs                # Constraint checks
│   └── ...                         # Other utilities
├── DTOs/
│   ├── ScheduleRequest.cs          # Input format
│   └── ScheduleResponse.cs         # Output format
├── example_input.json              # Test data
├── run.ps1                         # Start server
├── test-api.ps1                    # Test API
├── QUICKSTART.md                   # Quick start
├── README.md                       # Full docs
└── OVERVIEW.md                     # Technical details
```

## 🔍 Algorithm Implementation

The API implements your greedy scheduling algorithm with:
- ✅ Time preference bonuses
- ✅ Channel switch penalties
- ✅ Early termination penalties
- ✅ Minimum duration constraints
- ✅ Maximum consecutive genre constraints
- ✅ Priority time blocks
- ✅ Optimized program lookup with caching

## 🌟 Key Features

1. **RESTful API** - Standard HTTP POST endpoint
2. **JSON I/O** - Easy integration with any client
3. **Swagger UI** - Interactive documentation
4. **CORS Enabled** - Can be called from web apps
5. **Logging** - Built-in request/response logging
6. **Error Handling** - Graceful error responses
7. **Type Safety** - Strong typing with C#

## 💡 Usage Example

**Input** (example_input.json):
```json
{
  "opening_time": 540,
  "closing_time": 1080,
  "min_duration": 30,
  "channels": [...]
}
```

**Output**:
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

## 🔧 Requirements

- .NET 9.0 SDK (already installed on your machine)
- Any IDE or text editor
- PowerShell (for scripts, but not required)

## ✨ Next Steps

1. **Start the server**: `cd SchedulingAPI && ./run.ps1`
2. **Open Swagger**: Navigate to http://localhost:5191/swagger
3. **Test the API**: Run `./test-api.ps1` in a new terminal
4. **Try your own data**: Modify `example_input.json` or use files from `../data/input/`

## 📝 Notes

- The project is fully functional and tested
- It uses the same algorithm logic as your Python implementation
- All files are documented with comments
- The API is production-ready with proper error handling

## 🎓 Course Context

This project is part of your **Advanced Algorithms (AA_25-26)** course work at the University of Prishtina.

---

## 🎊 Summary

You now have a complete, production-ready ASP.NET Core Web API that:
- ✅ Accepts JSON scheduling data
- ✅ Runs your greedy scheduling algorithm
- ✅ Returns optimized schedules with scores
- ✅ Has interactive documentation
- ✅ Is fully documented and tested

**Start exploring by running `./run.ps1` in the SchedulingAPI directory!**

