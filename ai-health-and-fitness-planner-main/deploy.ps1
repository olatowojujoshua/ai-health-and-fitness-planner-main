# DEPLOY.ps1 - GitHub push helper (Windows PowerShell)
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoName,

    [string]$Description = "AI Health and Fitness Assistant",

    [switch]$Force
)

$ErrorActionPreference = "Stop"

# 1) GitHub Connection
$GitHubUser = "olatowojujoshua"
$remoteUrl = "https://github.com/$GitHubUser/$RepoName.git"

# Ensure git exists
git --version | Out-Null

# Ensure we are in a git repo
if (-not (Test-Path ".git")) {
    throw "No .git folder found. Run 'git init -b main' first (or clone the repo)."
}

# Set remote URL
git remote set-url origin $remoteUrl

# 2) Commit Changes (only if there are changes)
git add --all
$status = git status --porcelain
if ($status) {
    $msg = "🚀 Deploy: $RepoName"
    git commit -m $msg
} else {
    Write-Host "ℹ️ No changes to commit." -ForegroundColor Yellow
}

# 3) Push
if ($Force) {
    git push -u origin main --force
} else {
    git push -u origin main
}

Write-Host "✅ Pushed to: https://github.com/$GitHubUser/$RepoName" -ForegroundColor Green
Write-Host "Tip: Use -Force only when you really need it." -ForegroundColor Cyan
