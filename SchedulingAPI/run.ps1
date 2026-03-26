# Test the Scheduling API

Write-Host "Building the project..." -ForegroundColor Cyan
dotnet build

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "`nStarting the API server..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host "`nAPI Documentation available at: http://localhost:5191/swagger" -ForegroundColor Green
    Write-Host "API Endpoint: http://localhost:5191/api/schedule" -ForegroundColor Green
    Write-Host "`n" -ForegroundColor White
    
    dotnet run
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
}

