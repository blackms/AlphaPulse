@echo off
echo Creating GitHub release for v0.1.1...
echo.
echo Please provide your GitHub token when prompted.
echo You can get a token from: https://github.com/settings/tokens
echo.
set /p token="GitHub Token: "
powershell -ExecutionPolicy Bypass -File "create_release.ps1" -GitHubToken "%token%"
pause 