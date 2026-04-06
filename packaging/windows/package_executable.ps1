# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

param (
    [switch]$skipWinget = $false,
    [switch]$skipGvsbuild = $false,
    [switch]$skipTranslations = $false,
    [switch]$skipArchive = $false,
    [switch]$cliOnly = $false,
    [switch]$cleanGvsbuild = $false,
    [string]$extra = "nvidia",
    [string]$artifactFlavor = "vulkan"
)

$global:PYINSTALLER_VERSION = "6.18.0"
$global:GVSBUILD_VERSION = "2026.1.0"
$global:PYTHON_VERSION = "3.13.12"
$global:UV_VERSION = "0.10.0"

function Ask-YesNo {
    param([Parameter(Mandatory)] [string]$Question)

    while ($true) {
        $response = Read-Host "$Question (Y/N)"

        switch ($response.ToUpper()) {
            'Y' { return $true }
            'N' { return $false }
            default { Write-Host "Please enter Y or N." -ForegroundColor Yellow }
        }
    }
}

function Install-SystemDependencies {
    param([Parameter(Mandatory)] [boolean]$cliOnly)

    Write-Host "Installing system dependencies..."

    winget install --id Gyan.FFmpeg -e --source winget
    winget install --id Git.Git -e --source winget
    winget install --id=astral-sh.uv -e --source winget --version $global:UV_VERSION --force
    winget install --id=7zip.7zip -e --source winget

    if (-Not ($cliOnly)) {
        winget install --id MSYS2.MSYS2 -e --source winget
        winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        winget install --id Rustlang.Rustup -e --source winget
        winget install --id Microsoft.VCRedist.2013.x64  -e --source winget
        winget install --id Microsoft.VCRedist.2013.x86  -e --source winget
    }
}

function Build-SystemDependencies {
    param([Parameter(Mandatory)] [boolean]$clean)

    Write-Host "Building system dependencies..."

    uv venv --clear --python $global:PYTHON_VERSION venv_gtk_release
    .\venv_gtk_release\Scripts\Activate.ps1

    uv pip install gvsbuild==$global:GVSBUILD_VERSION
    uv pip install patch
    uv run --no-project python -m patch -p1 -d venv_gtk_release/lib/site-packages patches/gvsbuild_ffmpeg.patch
    @'
from pathlib import Path
import shutil

path = Path("venv_gtk_release/Lib/site-packages/gvsbuild/projects/pkgconf.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    'archive_url="https://distfiles.ariadne.space/pkgconf/pkgconf-{version}.tar.gz",',
    'archive_url="https://github.com/pkgconf/pkgconf/archive/refs/tags/pkgconf-{version}.tar.gz",',
)
text = text.replace(
    'hash="ab89d59810d9cad5dfcd508f25efab8ea0b1c8e7bad91c2b6351f13e6a5940d8",',
    'hash="79721badcad1987dead9c3609eb4877ab9b58821c06bdacb824f2c8897c11f2a",',
)
path.write_text(text, encoding="utf-8")

patch_src = Path("patches/gvsbuild_gobject_introspection_py313.patch")
patch_dst = Path("venv_gtk_release/Lib/site-packages/gvsbuild/patches/gobject-introspection/002-python313-msvccompiler.patch")
patch_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(patch_src, patch_dst)

project_path = Path("venv_gtk_release/Lib/site-packages/gvsbuild/projects/gobject_introspection.py")
project_text = project_path.read_text(encoding="utf-8")
needle = '                "001-incorrect-giscanner-path.patch",\n'
replacement = (
    '                "001-incorrect-giscanner-path.patch",\n'
    '                "002-python313-msvccompiler.patch",\n'
)
if "002-python313-msvccompiler.patch" not in project_text:
    project_text = project_text.replace(needle, replacement)
project_path.write_text(project_text, encoding="utf-8")

ffmpeg_project_path = Path("venv_gtk_release/Lib/site-packages/gvsbuild/projects/ffmpeg.py")
ffmpeg_project_text = ffmpeg_project_path.read_text(encoding="utf-8")
if "from pathlib import Path" not in ffmpeg_project_text:
    ffmpeg_project_text = ffmpeg_project_text.replace(
        "import os\n",
        "import os\nfrom pathlib import Path\n",
    )
ffmpeg_needle = """    def build(self):
        configuration = (
            "debug-optimized"
            if self.opts.release_configuration_is_actually_debug_optimized
            else self.opts.configuration
        )
"""
ffmpeg_replacement = """    def build(self):
        configuration = (
            "debug-optimized"
            if self.opts.release_configuration_is_actually_debug_optimized
            else self.opts.configuration
        )

        configure_path = Path(self.build_dir) / "configure"
        if configure_path.exists():
            configure_text = configure_path.read_text(encoding="utf-8")
            strict_probe = (
                '# Treat unrecognized flags as errors on MSVC\\n'
                'test_cpp_condition windows.h "_MSC_FULL_VER >= 193030705" &&\\n'
                '    check_cflags -options:strict\\n'
                'test_host_cpp_condition windows.h "_MSC_FULL_VER >= 193030705" &&\\n'
                '    check_host_cflags -options:strict\\n'
            )
            strict_probe_replacement = (
                '# gvsbuild workaround: recent MSVC accepts -options:strict, but FFmpeg configure\\n'
                '# still probes cl.exe with GCC-style -o arguments during compiler detection.\\n'
                '# Skip the strict-flag probe so the real MSVC toolchain build can proceed.\\n'
            )
            if strict_probe in configure_text and 'Skip the strict-flag probe' not in configure_text:
                configure_text = configure_text.replace(strict_probe, strict_probe_replacement)
            localized_probe = (
                '    elif VSLANG=1033 $_cc -nologo- 2>&1 | grep -q ^Microsoft || { $_cc -v 2>&1 | grep -q clang && $_cc -? > /dev/null 2>&1; }; then\\n'
            )
            localized_probe_replacement = (
                '    elif [ "$toolchain" = "msvc" ] || VSLANG=1033 $_cc -nologo- 2>&1 | grep -q ^Microsoft || { $_cc -v 2>&1 | grep -q clang && $_cc -? > /dev/null 2>&1; }; then\\n'
            )
            if localized_probe in configure_text and '[ "$toolchain" = "msvc" ]' not in configure_text:
                configure_text = configure_text.replace(localized_probe, localized_probe_replacement)
            localized_ident = (
                '        if VSLANG=1033 $_cc -nologo- 2>&1 | grep -q ^Microsoft; then\\n'
            )
            localized_ident_replacement = (
                '        if [ "$toolchain" = "msvc" ] || VSLANG=1033 $_cc -nologo- 2>&1 | grep -q ^Microsoft; then\\n'
            )
            if localized_ident in configure_text and 'if [ "$toolchain" = "msvc" ] || VSLANG=1033' not in configure_text:
                configure_text = configure_text.replace(localized_ident, localized_ident_replacement)
            if '[ "$toolchain" = "msvc" ]' in configure_text:
                configure_path.write_text(configure_text, encoding="utf-8")
"""
if "Skip the strict-flag probe" not in ffmpeg_project_text:
    ffmpeg_project_text = ffmpeg_project_text.replace(ffmpeg_needle, ffmpeg_replacement)
ffmpeg_project_path.write_text(ffmpeg_project_text, encoding="utf-8")
'@ | python -X utf8 -
    uv pip uninstall patch

    $cleanArgument = if ($clean) { '--clean' } else { '' }

    gvsbuild build `
        --configuration=release `
        $cleanArgument `
        --build-dir='./build_gtk_release' `
        --enable-gi `
        --py-wheel `
        gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugin-gtk4 gst-libav gst-python gettext

    deactivate
}

function Compile-Translations {
    Write-Host "Compiling translations..."
    # Adjust PATH to include gettext tools used in translation compile script
    $path_backup = $env:Path
    $gtk_bin_dir = (Resolve-Path ".\build_gtk_release\gtk\x64\release\bin").Path
    $env:Path = $gtk_bin_dir + ";" + $env:Path

    & ./translations/compile_po.ps1 --release

    $env:Path = $path_backup
}

function Download-ModelWeights {
    Write-Host "Downloading model weights..."

    function Is-Downloaded($file_name, $sha256) {
        $path = ".\model_weights\" + $file_name
        return (Test-Path $path) -And ((Get-FileHash -Algorithm SHA256 $path).Hash -eq $sha256)
    }

    function Download($url, $file_name, $sha256) {
        if (Is-Downloaded $file_name $sha256) {
            return
        }
        Invoke-WebRequest $url -OutFile (".\model_weights\" + $file_name)
        if (!(Is-Downloaded $file_name $sha256)) {
            Write-Warning "Error downloading " + $url
            exit 1
        }
    }

    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true' "lada_mosaic_detection_model_v2.pt" "056756fcab250bcdf0833e75aac33e2197b8809b0ab8c16e14722dcec94269b5"
    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true' "lada_mosaic_detection_model_v4_accurate.pt" "c244d7e49d8f88e264b8dc15f91fb21f5908ad8fb6f300b7bc88462d0801bc1f"
    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true' "lada_mosaic_detection_model_v4_fast.pt" "9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4"
    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true' "lada_mosaic_restoration_model_generic_v1.2.pth" "d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee"
    Download 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' "3rd_party\clean_youknow_video.pth" "5643ca297c13920b8ffd39a0d85296e494683a69e5e8204d662653d24c582766"
}

function Install-PythonDependencies {
    param(
        [Parameter(Mandatory)] [boolean]$cliOnly,
        [Parameter(Mandatory)] [string]$extra
    )

    Write-Host "Installing Python dependencies..."

    uv venv --clear --python $global:PYTHON_VERSION venv_release_win
    .\venv_release_win\Scripts\Activate.ps1

    uv sync --active --frozen --extra $extra --no-editable --no-install-project
    # Fix crash due to polars package requiring AVX512 CPU which isn't available on my build machine (use legacy version)
    # This dependency is not used by Lada. It gets pulled in by ultralytics which uses it outside of inferencing paths
    uv pip uninstall polars
    uv pip install polars-lts-cpu

    uv pip install --no-deps '.'
    if (-not $cliOnly) {
        uv pip install --force-reinstall (Resolve-Path ".\build_gtk_release\gtk\x64\release\python\pygobject*.whl").Path
        uv pip install --force-reinstall (Resolve-Path ".\build_gtk_release\gtk\x64\release\python\pycairo*.whl").Path
    }

    # pnnx is only needed at runtime for NCNN export paths, so keep it packaging-scoped.
    uv pip install pnnx

    # pin setuptools to fix build failure of gobject-introspection. Can be removed once https://github.com/wingtk/gvsbuild/pull/1715 is released
    uv pip install pyinstaller==$global:PYINSTALLER_VERSION "setuptools<81.0.0"

    uv pip install patch
    uv run --no-project python -m patch -p1 -d venv_release_win/lib/site-packages patches/increase_mms_time_limit.patch
    uv run --no-project python -m patch -p1 -d venv_release_win/lib/site-packages patches/remove_ultralytics_telemetry.patch
    uv run --no-project python -m patch -p1 -d venv_release_win/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
    uv pip uninstall patch

    deactivate
}

function Create-EXE {
    param([Parameter(Mandatory)] [boolean]$cliOnly,[Parameter(Mandatory)] [string]$extra)

    Write-Host "Creating executable for target: $extra..."

    .\venv_release_win\Scripts\Activate.ps1

    if (-not $cliOnly) {
        $release_dir = (Resolve-Path ".\build_gtk_release\gtk\x64\release").Path
        $env:Path = $release_dir + "\bin;" + $env:Path
        $env:LIB = $release_dir + "\lib;" + $env:LIB
        $env:INCLUDE = $release_dir + "\include;" + $release_dir + "\include\cairo;" + $release_dir + "\include\glib-2.0;" + $release_dir + "\include\gobject-introspection-1.0;" + $release_dir + "\lib\glib-2.0\include;" + $env:INCLUDE
    }

    $specArgs = @()
    if ($cliOnly) { $specArgs += '--cli-only' }
    $specArgs += "--extra=$extra"

    uv run --no-project pyinstaller --noconfirm ./packaging/windows/lada.spec -- $specArgs

    deactivate
}

function Create-7ZArchive {
    param(
        [Parameter(Mandatory)] [boolean]$cliOnly,
        [Parameter(Mandatory)] [string]$artifactFlavor
    )
    Write-Host "Creating 7z archive for artifact flavor: $artifactFlavor..."

    .\venv_release_win\Scripts\Activate.ps1
    $version = $(uv run --no-project python -c 'import lada; print(lada.VERSION)')
    deactivate

    $env:Path = ($env:Programfiles + "\7-Zip;") + $env:Path

    $archive_prefix = if ($cliOnly) { "lada-cli" } else { "lada" }
    $archive_base = "{0}-v{1}_windows_{2}" -f $archive_prefix,$version,$artifactFlavor
    $archive_path = "./dist/{0}.7z" -f $archive_base

    # Delete files from prior runs
    Get-ChildItem "./dist" -filter "*.7z*" | ForEach-Object {
        rm $_.FullName
    }

    # Split .7z archive into 2GB chunks so they can be uploaded to GitHub Releases
    7z.exe a -v1999m $archive_path "./dist/lada"

    # Create single-file .7z archive
    $single_chunk = (Get-ChildItem "./dist" -filter "*.7z*" | Where-Object Name -Match '\.7z.\d{3}$' | Measure-Object).Count -eq 1
    if ($single_chunk) {
        $old = "./dist/{0}.7z.001" -f $archive_base
        $new = $archive_path
        mv $old $new
    } else {
        7z.exe a $archive_path "./dist/lada"
    }

    Get-ChildItem "./dist" -filter "*.7z*" | Where-Object Name -Match '\.7z(.\d{3})?$' | ForEach-Object {
        $sha256 = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower()
        echo ($sha256 + " " + $_.Basename + $_.Extension) > ($_.FullName + ".sha256")
    }
}

function Check-ProjectRoot {
    if (!(Test-Path ".\pyproject.toml")) {
        Write-Warning "This script needs to be run from the project root directory."
        exit 1
    }
}

function Check-Extra {
    param([Parameter(Mandatory)] [string]$extra)
    if (-not ("nvidia", "intel" -contains $extra)) {
        Write-Warning "Currently only 'nvidia' or 'intel' extras are supported."
        exit 1
    }
}

function Check-ArtifactFlavor {
    param([Parameter(Mandatory)] [string]$artifactFlavor)
    if ([string]::IsNullOrWhiteSpace($artifactFlavor)) {
        Write-Warning "artifactFlavor must not be empty."
        exit 1
    }
}

# ---------------------
# EXECUTE PACKAGING STEPS
# ---------------------

$ErrorActionPreference = "Stop"

Check-ProjectRoot
Check-Extra $extra
Check-ArtifactFlavor $artifactFlavor

if (-not $skipWinget) {
    Install-SystemDependencies $cliOnly
    if (!(Ask-YesNo "Installing/Upgrading winget programs finished. Check the winget install output above. You may want to stop and restart this script in a new shell for certain installs/updates. Do you want to continue?")) {
        exit 0
    }
}
if (-not ($skipGvsbuild -Or $cliOnly)) {
    Build-SystemDependencies $cleanGvsbuild
}
if (-not ($skipTranslations -Or $cliOnly)) {
    Compile-Translations
}
Download-ModelWeights
Install-PythonDependencies $cliOnly $extra
Create-EXE $cliOnly $extra
if (-not $skipArchive) {
    Create-7ZArchive $cliOnly $artifactFlavor
}
