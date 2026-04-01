param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl,

    [string]$RemoteName = "origin",

    [string]$BranchName = "main"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    git init -b $BranchName
}

$existingRemote = git remote
if ($existingRemote -contains $RemoteName) {
    git remote set-url $RemoteName $RemoteUrl
} else {
    git remote add $RemoteName $RemoteUrl
}

git branch --set-upstream-to="$RemoteName/$BranchName" $BranchName 2>$null | Out-Null

Write-Host "Remote configured:"
Write-Host "  Name: $RemoteName"
Write-Host "  URL : $RemoteUrl"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  git add ."
Write-Host "  git commit -m 'Initial cloud workflow setup'"
Write-Host "  git push -u $RemoteName $BranchName"

