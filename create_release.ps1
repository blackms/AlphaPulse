# PowerShell script to create GitHub release for v0.1.1
# Usage: .\create_release.ps1 [GITHUB_TOKEN]

param(
    [string]$GitHubToken = $env:GITHUB_TOKEN
)

if (-not $GitHubToken) {
    Write-Host "GitHub token not provided. Please provide a token as parameter or set GITHUB_TOKEN environment variable."
    Write-Host "Usage: .\create_release.ps1 [GITHUB_TOKEN]"
    exit 1
}

$owner = "blackms"
$repo = "AlphaPulse"
$tag = "v0.1.1"
$releaseName = "v0.1.1"
$body = @"
## Release v0.1.1

### Changed
- Refactored backtester to use new `alpha_pulse/agents` module instead of deprecated `src/agents`.
- Removed the old `src/agents` directory and all legacy agent code.
- Confirmed all documentation and diagrams are up-to-date after agents module cleanup.

### Technical Details
This release includes a major code cleanup that:
- Migrates the backtesting system from the old `src/agents` implementation to the new `alpha_pulse/agents` module
- Removes legacy agent code that was no longer maintained
- Ensures all documentation and architecture diagrams remain accurate
- Improves code maintainability and reduces technical debt

### Breaking Changes
None - this is a refactoring release that maintains backward compatibility.
"@

$headers = @{
    "Authorization" = "token $GitHubToken"
    "Accept" = "application/vnd.github.v3+json"
}

$body = $body -replace "`n", "`n"

$releaseData = @{
    tag_name = $tag
    name = $releaseName
    body = $body
    draft = $false
    prerelease = $false
} | ConvertTo-Json -Depth 10

$uri = "https://api.github.com/repos/$owner/$repo/releases"

try {
    Write-Host "Creating release $tag for $owner/$repo..."
    $response = Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -Body $releaseData -ContentType "application/json"
    Write-Host "Release created successfully!"
    Write-Host "Release URL: $($response.html_url)"
} catch {
    Write-Host "Error creating release: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody"
    }
} 