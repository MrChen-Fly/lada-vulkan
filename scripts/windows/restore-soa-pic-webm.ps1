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
    return (Join-Path $PSScriptRoot "restore_soa_pic_webm_state.json")
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

function Test-WebmContainerSignature {
    param([Parameter(Mandatory)] [string]$Path)

    $stream = $null
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        $header = New-Object byte[] 4
        $bytesRead = $stream.Read($header, 0, $header.Length)
        return (
            $bytesRead -eq 4 -and
            $header[0] -eq 0x1A -and
            $header[1] -eq 0x45 -and
            $header[2] -eq 0xDF -and
            $header[3] -eq 0xA3
        )
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
    }
}

function Test-OutputFileReady {
    param([Parameter(Mandatory)] [string]$OutputPath)

    if (-not (Test-Path -LiteralPath $OutputPath)) {
        return $false
    }

    $outputExtension = [System.IO.Path]::GetExtension($OutputPath).ToLowerInvariant()
    if ($outputExtension -eq ".webm") {
        return (Test-WebmContainerSignature -Path $OutputPath)
    }

    return $true
}

function Get-DefaultWebmEncodingArgs {
    # Use a WebM-compatible encoder by default so the final ".webm" output is
    # a real WebM container instead of a renamed MP4 file.
    return @("--encoding-preset", "av1-cpu-uhq")
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
        @((Join-Path $ResolvedProjectRoot ".venv\Scripts\python.exe"), "-m", "lada.cli.main"),
        @((Join-Path $ResolvedProjectRoot "venv_release_win\Scripts\python.exe"), "-m", "lada.cli.main"),
        (Join-Path $ResolvedProjectRoot ".venv\Scripts\lada-cli.exe"),
        (Join-Path $ResolvedProjectRoot "venv_release_win\Scripts\lada-cli.exe"),
        (Join-Path $ResolvedProjectRoot "dist\lada\lada-cli.exe")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -is [System.Array]) {
            if (Test-Path -LiteralPath $candidate[0]) {
                return $candidate
            }
            continue
        }
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

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory)] [int]$ProcessId,
        [switch]$IncludeRoot = $true
    )

    if ($ProcessId -le 0) {
        return
    }

    $descendantIds = @()
    $pendingParents = [System.Collections.Generic.Queue[int]]::new()
    $pendingParents.Enqueue($ProcessId)
    while ($pendingParents.Count -gt 0) {
        $parentId = $pendingParents.Dequeue()
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $parentId" -ErrorAction SilentlyContinue)
        foreach ($child in $children) {
            $childId = [int]$child.ProcessId
            if ($childId -le 0 -or $descendantIds -contains $childId) {
                continue
            }
            $descendantIds += $childId
            $pendingParents.Enqueue($childId)
        }
    }

    $taskKill = Get-Command "taskkill.exe" -ErrorAction SilentlyContinue
    $rootProcess = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($IncludeRoot -and $null -ne $taskKill -and $null -ne $rootProcess) {
        & $taskKill.Source /PID $ProcessId /T /F *> $null
    }
    elseif ($IncludeRoot -and $null -ne $rootProcess) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }

    foreach ($childId in ($descendantIds | Sort-Object -Descending)) {
        Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue
    }
}

function ConvertTo-ProcessArguments {
    param([Parameter(Mandatory)] [string[]]$Arguments)

    $escapedArguments = foreach ($argument in $Arguments) {
        $value = [string]$argument
        if ($value.Length -eq 0) {
            '""'
            continue
        }
        if ($value -notmatch '[\s"]') {
            $value
            continue
        }

        $quotedValue = $value -replace '(\\*)"', '$1$1\"'
        $quotedValue = $quotedValue -replace '(\\+)$', '$1$1'
        '"' + $quotedValue + '"'
    }

    return [string]::Join(' ', $escapedArguments)
}

function Start-ConsoleCancelMonitor {
    $monitorState = @{
        TreatCtrlCAsInputEnabled = $false
        OriginalTreatCtrlCAsInput = $false
        CancelEventInfo = $null
        CancelHandler = $null
    }

    try {
        $monitorState.OriginalTreatCtrlCAsInput = [System.Console]::TreatControlCAsInput
        [System.Console]::TreatControlCAsInput = $true
        $monitorState.TreatCtrlCAsInputEnabled = $true
    }
    catch {
        # Some non-interactive hosts do not expose a readable console input
        # buffer. In that case we keep the legacy cancel event fallback only.
    }

    $monitorState.CancelHandler = [System.ConsoleCancelEventHandler]{
        param($sender, $eventArgs)

        $eventArgs.Cancel = $true
        Request-LadaBatchCancellation
    }
    $monitorState.CancelEventInfo = [System.Console].GetEvent("CancelKeyPress")
    $monitorState.CancelEventInfo.AddEventHandler($null, $monitorState.CancelHandler)

    return $monitorState
}

function Test-ConsoleCancelRequested {
    param([Parameter(Mandatory)] [hashtable]$MonitorState)

    if (-not $MonitorState.TreatCtrlCAsInputEnabled) {
        return $false
    }

    try {
        while ([System.Console]::KeyAvailable) {
            $keyInfo = [System.Console]::ReadKey($true)
            $isCtrlC = (
                $keyInfo.Key -eq [System.ConsoleKey]::C -and
                (($keyInfo.Modifiers -band [System.ConsoleModifiers]::Control) -ne 0)
            )
            if ($isCtrlC) {
                return $true
            }
        }
    }
    catch {
        return $false
    }

    return $false
}

function Stop-ConsoleCancelMonitor {
    param([hashtable]$MonitorState)

    if ($null -eq $MonitorState) {
        return
    }

    if (($null -ne $MonitorState.CancelEventInfo) -and ($null -ne $MonitorState.CancelHandler)) {
        $MonitorState.CancelEventInfo.RemoveEventHandler($null, $MonitorState.CancelHandler)
    }

    if ($MonitorState.TreatCtrlCAsInputEnabled) {
        try {
            [System.Console]::TreatControlCAsInput = [bool]$MonitorState.OriginalTreatCtrlCAsInput
        }
        catch {
        }
    }
}

function Request-LadaBatchCancellation {
    if (-not $script:RestoreSoaPicWebmCancelled) {
        $script:RestoreSoaPicWebmCancelled = $true
    }

    if (-not $script:RestoreSoaPicWebmCancellationNoticeShown) {
        $script:RestoreSoaPicWebmCancellationNoticeShown = $true
        Write-Warning "Ctrl+C received, stopping lada-cli..."
    }

    if ($script:RestoreSoaPicWebmProcessId -gt 0) {
        Stop-ProcessTree -ProcessId $script:RestoreSoaPicWebmProcessId
    }
}

function Invoke-LadaCliBatch {
    param(
        [Parameter(Mandatory)] [string[]]$Command,
        [Parameter(Mandatory)] [string[]]$Arguments,
        [Parameter(Mandatory)] [string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    $ladaProcess = $null
    $ladaExitCode = 0
    $cancelRequested = $false
    $cancelMonitor = $null
    $hadSvtLog = Test-Path Env:SVT_LOG
    $originalSvtLog = if ($hadSvtLog) { $env:SVT_LOG } else { $null }
    $script:RestoreSoaPicWebmCancelled = $false
    $script:RestoreSoaPicWebmProcessId = 0
    $script:RestoreSoaPicWebmCancellationNoticeShown = $false
    try {
        $commandArgs = @()
        if ($Command.Count -gt 1) {
            $commandArgs = $Command[1..($Command.Count - 1)]
        }

        $processArguments = ConvertTo-ProcessArguments -Arguments @($commandArgs + $Arguments)
        if (-not $hadSvtLog) {
            $env:SVT_LOG = "0"
        }

        $cancelMonitor = Start-ConsoleCancelMonitor

        $processStartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processStartInfo.FileName = $Command[0]
        $processStartInfo.Arguments = $processArguments
        $processStartInfo.WorkingDirectory = $WorkingDirectory
        $processStartInfo.UseShellExecute = $false
        $processStartInfo.RedirectStandardOutput = $false
        $processStartInfo.RedirectStandardError = $false
        $processStartInfo.CreateNoWindow = $false

        $ladaProcess = New-Object System.Diagnostics.Process
        $ladaProcess.StartInfo = $processStartInfo
        $null = $ladaProcess.Start()
        $script:RestoreSoaPicWebmProcessId = $ladaProcess.Id

        while (-not $ladaProcess.WaitForExit(200)) {
            if (Test-ConsoleCancelRequested -MonitorState $cancelMonitor) {
                Request-LadaBatchCancellation
            }

            if ($script:RestoreSoaPicWebmCancelled) {
                $cancelRequested = $true
                $null = $ladaProcess.WaitForExit(5000)
                break
            }
        }

        $cancelRequested = $script:RestoreSoaPicWebmCancelled
        $ladaExitCode = if ($cancelRequested) { 130 } else { $ladaProcess.ExitCode }
    }
    finally {
        Stop-ConsoleCancelMonitor -MonitorState $cancelMonitor
        if ($hadSvtLog) {
            $env:SVT_LOG = $originalSvtLog
        }
        else {
            Remove-Item Env:SVT_LOG -ErrorAction SilentlyContinue
        }
        if ($null -ne $ladaProcess) {
            try {
                if (-not $ladaProcess.HasExited) {
                    Stop-ProcessTree -ProcessId $ladaProcess.Id
                }
                else {
                    Stop-ProcessTree -ProcessId $ladaProcess.Id -IncludeRoot:$false
                }
            }
            finally {
                $ladaProcess.Dispose()
            }
        }
        $script:RestoreSoaPicWebmCancelled = $false
        $script:RestoreSoaPicWebmProcessId = 0
        $script:RestoreSoaPicWebmCancellationNoticeShown = $false
        Pop-Location
    }

    return $ladaExitCode
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
    Get-DefaultStatePath
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
$staleOutputPaths = New-Object System.Collections.Generic.List[string]
foreach ($file in $sourceFiles) {
    $relativePath = Get-RelativePath -BasePath $SourceRoot -TargetPath $file.FullName
    $finalOutputPath = Join-Path $resolvedOutputRoot $relativePath
    $backupPath = Join-Path $resolvedBackupRoot $relativePath

    if ((-not $ForceReconvert) -and (Test-Path -LiteralPath $finalOutputPath)) {
        if (Test-OutputFileReady -OutputPath $finalOutputPath) {
            $readyOutputCount++
        }
        else {
            $staleOutputPaths.Add($finalOutputPath)
        }
    }
    if ((-not $ForceBackup) -and (Test-BackupFileReady -SourceFile $file -BackupPath $backupPath)) {
        $readyBackupCount++
    }
}

$staleOutputPaths = @($staleOutputPaths | Sort-Object -Unique)
if ($staleOutputPaths.Count -gt 0) {
    Write-Warning "Found $($staleOutputPaths.Count) stale output file(s) with mismatched container signatures. They will be regenerated."
    if (-not $DryRun) {
        foreach ($staleOutputPath in $staleOutputPaths) {
            if (-not (Test-IsPathInside -ParentPath $resolvedOutputRoot -ChildPath $staleOutputPath)) {
                throw "Refusing to remove stale output outside output root: $staleOutputPath"
            }
            Remove-Item -LiteralPath $staleOutputPath -Force
        }
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
    "--working-output-extension", ".working.webm",
    "--output-file-pattern", "{orig_file_name}.webm"
)

$explicitEncodingOverride = $false
$explicitFp16Override = $false
foreach ($arg in $ExtraLadaArgs) {
    if ($arg -in @("--encoding-preset", "--encoder", "--encoder-options")) {
        $explicitEncodingOverride = $true
    }
    if ($arg -in @("--fp16", "--no-fp16")) {
        $explicitFp16Override = $true
    }
}
if (-not $explicitEncodingOverride) {
    $cliArgs += Get-DefaultWebmEncodingArgs
}
if ((-not $explicitFp16Override) -and $Device.ToLowerInvariant().StartsWith("vulkan")) {
    $cliArgs += "--fp16"
}

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
if ($staleOutputPaths.Count -gt 0) {
    Write-Host "Stale output: $($staleOutputPaths.Count) invalid file(s) will be regenerated"
}
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

$ladaBatchExitCode = Invoke-LadaCliBatch -Command $resolvedLadaCommand -Arguments $cliArgs -WorkingDirectory $resolvedProjectRoot
$global:LASTEXITCODE = $ladaBatchExitCode

if ($ladaBatchExitCode -eq 130) {
    Write-Warning "lada-cli cancelled by Ctrl+C."
    return
}

if ($ladaBatchExitCode -ne 0) {
    throw "lada-cli exited with code $ladaBatchExitCode"
}
