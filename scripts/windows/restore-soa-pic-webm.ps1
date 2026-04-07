# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

[CmdletBinding()]
param(
    [string]$SourceRoot = "D:\Games\SOA-electron-V88\assets\standalone_content\pic",
    [string]$ProjectRoot,
    [string]$BackupRoot,
    [string]$OutputRoot,
    [string]$StatePath,
    [string]$Device = "vulkan:0",
    [string[]]$LadaCommand = @(),
    [string[]]$ExtraLadaArgs = @(),
    [switch]$ForceBackup,
    [switch]$ForceReconvert,
    [switch]$ResetState,
    [switch]$DryRun,
    [int]$MaxFiles = 0,
    [int]$RetryCount = 2,
    [int]$RetryDelaySeconds = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-DefaultProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Get-DefaultStatePath {
    param([Parameter(Mandatory)] [string]$ResolvedProjectRoot)

    return (Join-Path $ResolvedProjectRoot ".helloagents\tmp\restore_soa_pic_webm_state.json")
}

function Get-FullPath {
    param([Parameter(Mandatory)] [string]$Path)

    return [System.IO.Path]::GetFullPath($Path)
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory)] [string]$BasePath,
        [Parameter(Mandatory)] [string]$TargetPath
    )

    $normalizedBasePath = Get-FullPath $BasePath
    if (-not $normalizedBasePath.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $normalizedBasePath += [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = [System.Uri]::new($normalizedBasePath)
    $targetUri = [System.Uri]::new((Get-FullPath $TargetPath))
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)

    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Test-IsPathInside {
    param(
        [Parameter(Mandatory)] [string]$ParentPath,
        [Parameter(Mandatory)] [string]$ChildPath
    )

    $parent = Get-FullPath $ParentPath
    $child = Get-FullPath $ChildPath
    if ($child.Equals($parent, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    if (-not $parent.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $parent += [System.IO.Path]::DirectorySeparatorChar
    }

    return $child.StartsWith($parent, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-BackupFileReady {
    param(
        [Parameter(Mandatory)] [System.IO.FileInfo]$SourceFile,
        [Parameter(Mandatory)] [string]$BackupPath
    )

    if (-not (Test-Path -LiteralPath $BackupPath)) {
        return $false
    }

    try {
        $backupFile = Get-Item -LiteralPath $BackupPath -ErrorAction Stop
    }
    catch {
        return $false
    }

    return (
        $backupFile.Length -eq $SourceFile.Length -and
        $backupFile.LastWriteTimeUtc -eq $SourceFile.LastWriteTimeUtc
    )
}

function Resolve-LadaCliCommand {
    param(
        [Parameter(Mandatory)] [string]$ResolvedProjectRoot,
        [string[]]$Command
    )

    if ($Command.Count -gt 0) {
        return $Command
    }

    $candidates = @(
        (Join-Path $ResolvedProjectRoot ".venv\Scripts\lada-cli.exe"),
        (Join-Path $ResolvedProjectRoot "venv_release_win\Scripts\lada-cli.exe"),
        (Join-Path $ResolvedProjectRoot "dist\lada\lada-cli.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return @($candidate)
        }
    }

    $ladaCli = Get-Command "lada-cli" -ErrorAction SilentlyContinue
    if ($null -ne $ladaCli) {
        return @($ladaCli.Source)
    }

    $uv = Get-Command "uv" -ErrorAction SilentlyContinue
    if ($null -ne $uv) {
        return @($uv.Source, "run", "--extra", "nvidia", "lada-cli")
    }

    throw "Unable to resolve a lada CLI command. Pass -LadaCommand explicitly."
}

function Invoke-LadaCliBatch {
    param(
        [Parameter(Mandatory)] [string[]]$Command,
        [Parameter(Mandatory)] [string[]]$Arguments,
        [Parameter(Mandatory)] [string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    try {
        $commandArgs = @()
        if ($Command.Count -gt 1) {
            $commandArgs = $Command[1..($Command.Count - 1)]
        }
        & $Command[0] @($commandArgs + $Arguments)
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
    throw "Source root does not exist: $SourceRoot"
}

$resolvedProjectRoot = if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    Get-DefaultProjectRoot
}
else {
    Get-FullPath $ProjectRoot
}

$resolvedBackupRoot = if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
    Join-Path $resolvedProjectRoot "pic_bak"
}
else {
    Get-FullPath $BackupRoot
}

$resolvedOutputRoot = if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    Join-Path $resolvedProjectRoot "pic_dist"
}
else {
    Get-FullPath $OutputRoot
}

$resolvedStatePath = if ([string]::IsNullOrWhiteSpace($StatePath)) {
    Get-DefaultStatePath -ResolvedProjectRoot $resolvedProjectRoot
}
else {
    Get-FullPath $StatePath
}

if ($RetryCount -lt 0) {
    throw "RetryCount must be >= 0"
}
if ($RetryDelaySeconds -lt 0) {
    throw "RetryDelaySeconds must be >= 0"
}
if ($MaxFiles -lt 0) {
    throw "MaxFiles must be >= 0"
}
if (Test-IsPathInside -ParentPath $SourceRoot -ChildPath $resolvedBackupRoot) {
    throw "Backup root must not be inside source root: $resolvedBackupRoot"
}
if (Test-IsPathInside -ParentPath $SourceRoot -ChildPath $resolvedOutputRoot) {
    throw "Output root must not be inside source root: $resolvedOutputRoot"
}

$resolvedLadaCommand = Resolve-LadaCliCommand -ResolvedProjectRoot $resolvedProjectRoot -Command $LadaCommand
$stateDirectory = Split-Path -Parent $resolvedStatePath
if (-not [string]::IsNullOrWhiteSpace($stateDirectory)) {
    New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
}
if ($ResetState -and (Test-Path -LiteralPath $resolvedStatePath)) {
    Remove-Item -LiteralPath $resolvedStatePath -Force
}

$sourceFiles = @(Get-ChildItem -LiteralPath $SourceRoot -Recurse -File -Filter "*.webm" | Sort-Object FullName)
if ($MaxFiles -gt 0) {
    $sourceFiles = @($sourceFiles | Select-Object -First $MaxFiles)
}
if ($sourceFiles.Count -eq 0) {
    throw "No .webm files found under $SourceRoot"
}

$readyOutputCount = 0
$readyBackupCount = 0
foreach ($file in $sourceFiles) {
    $relativePath = Get-RelativePath -BasePath $SourceRoot -TargetPath $file.FullName
    $finalOutputPath = Join-Path $resolvedOutputRoot $relativePath
    $backupPath = Join-Path $resolvedBackupRoot $relativePath

    if ((-not $ForceReconvert) -and (Test-Path -LiteralPath $finalOutputPath)) {
        $readyOutputCount++
    }
    if ((-not $ForceBackup) -and (Test-BackupFileReady -SourceFile $file -BackupPath $backupPath)) {
        $readyBackupCount++
    }
}

$pendingOutputCount = if ($ForceReconvert) { $sourceFiles.Count } else { $sourceFiles.Count - $readyOutputCount }
$pendingBackupCount = if ($ForceBackup) { $sourceFiles.Count } else { $sourceFiles.Count - $readyBackupCount }

$cliArgs = @(
    "--input", (Get-FullPath $SourceRoot),
    "--output", $resolvedOutputRoot,
    "--device", $Device,
    "--recursive",
    "--preserve-relative-paths",
    "--backup-root", $resolvedBackupRoot,
    "--batch-state-path", $resolvedStatePath,
    "--retry-failed-first",
    "--retry-count", [string]$RetryCount,
    "--retry-delay-seconds", [string]$RetryDelaySeconds,
    "--working-output-extension", ".mp4",
    "--output-file-pattern", "{orig_file_name}.webm"
)

if ($ForceBackup) {
    $cliArgs += "--force-backup"
}
if ($ForceReconvert) {
    $cliArgs += "--force-reconvert"
}
if ($DryRun) {
    $cliArgs += "--dry-run"
}
if ($MaxFiles -gt 0) {
    $cliArgs += @("--max-files", [string]$MaxFiles)
}
if ($ExtraLadaArgs.Count -gt 0) {
    $cliArgs += $ExtraLadaArgs
}

Write-Host "Source root : $(Get-FullPath $SourceRoot)"
Write-Host "Project root: $resolvedProjectRoot"
Write-Host "Backup root : $resolvedBackupRoot"
Write-Host "Output root : $resolvedOutputRoot"
Write-Host "State file  : $resolvedStatePath"
Write-Host "Device      : $Device"
Write-Host "Max files   : $(if ($MaxFiles -gt 0) { $MaxFiles } else { 'all' })"
Write-Host "Files found : $($sourceFiles.Count)"
Write-Host "Resume      : backup ready $readyBackupCount/$($sourceFiles.Count), output ready $readyOutputCount/$($sourceFiles.Count)"
Write-Host "Lada command: $($resolvedLadaCommand -join ' ')"
if ($DryRun) {
    Write-Host "Mode        : DryRun"
}
Write-Host ""

if ((-not $DryRun) -and $pendingOutputCount -le 0 -and $pendingBackupCount -le 0) {
    Write-Host "Nothing pending. Backups and outputs are already complete."
    exit 0
}

Write-Host "Running internal lada-cli batch mode..."
Write-Host ""

Invoke-LadaCliBatch -Command $resolvedLadaCommand -Arguments $cliArgs -WorkingDirectory $resolvedProjectRoot
