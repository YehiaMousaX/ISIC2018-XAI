param(
    [Parameter(Position=0)]
    [string]$Note = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bash = Get-Command bash -ErrorAction SilentlyContinue

if (-not $bash) {
    Write-Error "bash was not found in PATH. Install Git Bash and retry."
    exit 1
}

Push-Location $scriptDir
try {
    if ([string]::IsNullOrWhiteSpace($Note)) {
        & bash scripts/run.sh
    }
    else {
        & bash scripts/run.sh "$Note"
    }

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
