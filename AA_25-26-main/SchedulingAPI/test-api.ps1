# Test the API with example data
$url = "http://localhost:5191/api/schedule"
$jsonBody = Get-Content "example_input.json" -Raw

Write-Host "Testing API at $url" -ForegroundColor Cyan
Write-Host "Sending request..." -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $jsonBody
    
    Write-Host "`nSuccess!" -ForegroundColor Green
    Write-Host "`nTotal Score: $($response.total_score)" -ForegroundColor Cyan
    Write-Host "`nScheduled Programs:" -ForegroundColor Cyan
    
    $response.scheduled_programs | ForEach-Object {
        Write-Host "  Program: $($_.program_id) | Channel: $($_.channel_id) | Time: $($_.start)-$($_.end)" -ForegroundColor White
    }
    
    Write-Host "`nFull Response:" -ForegroundColor Cyan
    $response | ConvertTo-Json -Depth 10 | Write-Host
    
} catch {
    Write-Host "`nError: $_" -ForegroundColor Red
    Write-Host "Make sure the API server is running (run './run.ps1' in another terminal)" -ForegroundColor Yellow
}

