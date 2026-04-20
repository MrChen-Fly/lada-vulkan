param(
    [bool]$WithVulkan = $true,
    [bool]$BuildLocalRuntime = $true,
    [bool]$WithNcnnBenchmark = $false
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Join-Path $repoRoot "native/vulkan/ncnn_runtime"
$buildDir = Join-Path $projectRoot "build"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    throw "Visual Studio Build Tools with C++ workload not found."
}

$vsDevShellModule = Join-Path $vsInstall "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
if (-not (Test-Path $vsDevShellModule)) {
    throw "Microsoft.VisualStudio.DevShell.dll not found."
}

Import-Module $vsDevShellModule -Force
Enter-VsDevShell -VsInstallPath $vsInstall -Arch amd64 -HostArch amd64 | Out-Null

$ncnnSourceDir = Join-Path $repoRoot ".helloagents\tmp\ncnn-src"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

$vulkanFlag = if ($WithVulkan) { "ON" } else { "OFF" }
$localRuntimeFlag = if ($BuildLocalRuntime) { "ON" } else { "OFF" }
$benchmarkFlag = if ($WithNcnnBenchmark) { "ON" } else { "OFF" }

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$configureArgs = @(
    "-S", $projectRoot,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLADA_NCNN_WITH_VULKAN=$vulkanFlag",
    "-DLADA_NCNN_BUILD_LOCAL_RUNTIME=$localRuntimeFlag",
    "-DNCNN_BENCHMARK=$benchmarkFlag"
)
if (Test-Path $ncnnSourceDir) {
    $configureArgs += "-DLADA_NCNN_SOURCE_DIR=$ncnnSourceDir"
}
if (Test-Path $pythonExe) {
    $configureArgs += "-DPython3_EXECUTABLE=$pythonExe"
}

& cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    throw "cmake configure failed with exit code $LASTEXITCODE."
}

$targets = @("lada_ncnn_deformconv_bench")
if ($BuildLocalRuntime) {
    $targets += "lada_ncnn_local_runtime"
}

$buildArgs = @(
    "--build", $buildDir,
    "--config", "Release",
    "--target"
) + $targets

& cmake @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "cmake build failed with exit code $LASTEXITCODE."
}
