param(
    [bool]$WithVulkan = $true,
    [bool]$BuildLocalRuntime = $true,
    [bool]$WithNcnnBenchmark = $false
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Join-Path $repoRoot "native/ncnn_vulkan_runtime"
$buildDir = Join-Path $projectRoot "build"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    throw "Visual Studio Build Tools with C++ workload not found."
}

$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
$ncnnSourceDir = Join-Path $repoRoot ".helloagents\tmp\ncnn-src"
$ncnnSourceArg = if (Test-Path $ncnnSourceDir) {
    " -DLADA_NCNN_SOURCE_DIR=`"$ncnnSourceDir`""
} else {
    ""
}
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonArg = if (Test-Path $pythonExe) { " -DPython3_EXECUTABLE=`"$pythonExe`"" } else { "" }

$vulkanFlag = if ($WithVulkan) { "ON" } else { "OFF" }
$localRuntimeFlag = if ($BuildLocalRuntime) { "ON" } else { "OFF" }
$benchmarkFlag = if ($WithNcnnBenchmark) { "ON" } else { "OFF" }

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$configure = @(
    "call `"$vsDevCmd`" -host_arch=amd64 -arch=amd64",
    "cmake -S `"$projectRoot`" -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLADA_NCNN_WITH_VULKAN=$vulkanFlag -DLADA_NCNN_BUILD_LOCAL_RUNTIME=$localRuntimeFlag -DNCNN_BENCHMARK=$benchmarkFlag$ncnnSourceArg$pythonArg"
) -join " && "

$targets = @("lada_ncnn_deformconv_bench")
if ($BuildLocalRuntime) {
    $targets += "lada_ncnn_local_runtime"
}
$targetList = $targets -join " "

$build = @(
    "call `"$vsDevCmd`" -host_arch=amd64 -arch=amd64",
    "cmake --build `"$buildDir`" --target $targetList --config Release"
) -join " && "

cmd /c $configure
cmd /c $build
