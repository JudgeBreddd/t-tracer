<#
  T-Tracer - Windows installer.

  Brings a bare Windows machine to a working app: Python if missing, both
  virtual environments, every Python dependency, the ControlNet lineart model,
  and Start Menu / Desktop shortcuts.

  Safe to re-run. Every step checks whether it is already done.

  Run directly:
      powershell -ExecutionPolicy Bypass -File install.ps1
  or let the packaged Setup.exe call it (see installer.iss).
#>

[CmdletBinding()]
param(
    # Where the app is installed. Defaults to the folder this script sits in,
    # which is what the packaged installer wants.
    [string]$AppDir = $PSScriptRoot,
    [switch]$NoShortcuts
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # or Invoke-WebRequest crawls

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

$ProjectDir  = Split-Path -Parent $AppDir
$LineartVenv = Join-Path $env:LOCALAPPDATA 't-tracer\lineart-venv'

Write-Host ''
Write-Host '  T-Tracer - Setup' -ForegroundColor White
Write-Host '  ------------------------------'
Write-Host ''

# ---------------------------------------------------------------- Python ---
# 3.12 specifically, not "latest". A brand-new Python release routinely has no
# wheels for numpy/scipy/opencv/torch for weeks, and the Linux install already
# died exactly that way on 3.14 (pydantic-core tried to build through Rust).
$PyVersion = '3.12.8'

function Find-Python {
    foreach ($cmd in @('py -3.12', 'py -3', 'python')) {
        $parts = $cmd.Split(' ')
        $exe = Get-Command $parts[0] -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        try {
            $args = @()
            if ($parts.Count -gt 1) { $args += $parts[1] }
            $v = & $exe.Source @args -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
        } catch { continue }
        if (-not $v) { continue }
        $mm = $v.Trim().Split('.')
        if ([int]$mm[0] -eq 3 -and [int]$mm[1] -ge 10 -and [int]$mm[1] -le 13) {
            return @{ Exe = $exe.Source; Args = $args; Version = $v.Trim() }
        }
    }
    return $null
}

$py = Find-Python
if ($py) {
    Say "Found Python $($py.Version)"
} else {
    Say "Installing Python $PyVersion (none suitable found)"
    $installer = Join-Path $env:TEMP "python-$PyVersion-amd64.exe"
    $url = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-amd64.exe"
    try {
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    } catch {
        Die "Could not download Python from python.org. Check the connection, or install Python 3.12 yourself and re-run this."
    }
    # Per-user install: no admin rights needed, which matters because the whole
    # point is that someone can run this on a machine they do not administer.
    $p = Start-Process -FilePath $installer -Wait -PassThru -ArgumentList @(
        '/quiet', 'InstallAllUsers=0', 'PrependPath=1',
        'Include_pip=1', 'Include_launcher=1', 'Include_test=0'
    )
    if ($p.ExitCode -ne 0) { Die "The Python installer exited with code $($p.ExitCode)." }
    Remove-Item $installer -ErrorAction SilentlyContinue
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $py = Find-Python
    if (-not $py) { Die 'Python installed but is still not on PATH. Open a new terminal and re-run this script.' }
    Say "Python $($py.Version) installed"
}

function Invoke-Py { & $py.Exe @($py.Args) @args }

# ------------------------------------------------------------ main venv ---
$MainVenv = Join-Path $ProjectDir '.venv'
$MainPy   = Join-Path $MainVenv 'Scripts\python.exe'
if (-not (Test-Path $MainPy)) {
    Say 'Creating the tracing environment'
    Invoke-Py -m venv $MainVenv
}
Say 'Installing tracing + app dependencies'
& $MainPy -m pip install --upgrade pip --quiet
# --only-binary=pydantic-core is scoped, not blanket: it is the one dependency
# that builds through Rust, and a missing wheel there produces a 125-line cargo
# error instead of a readable failure. A blanket flag would wrongly reject
# proxy_tools, a pure-Python sdist pywebview needs.
& $MainPy -m pip install --only-binary=pydantic-core --quiet `
    -r (Join-Path $ProjectDir 'requirements.txt') `
    -r (Join-Path $AppDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Die 'Dependency install failed. See the messages above.' }

# ------------------------------------------------------- annotator venv ---
# Separate on purpose. Torch is ~2 GB and only the `nested` and `composite`
# strategies need it; keeping it apart means the app starts fast and a broken
# torch cannot take plain tracing down with it.
$LinPy = Join-Path $LineartVenv 'Scripts\python.exe'
if (-not (Test-Path $LinPy)) {
    Say 'Creating the annotator environment'
    New-Item -ItemType Directory -Force -Path (Split-Path $LineartVenv) | Out-Null
    Invoke-Py -m venv $LineartVenv
}
Say 'Installing PyTorch and the annotator (this is the slow part, ~2 GB)'
& $LinPy -m pip install --upgrade pip --quiet
$cuda = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($cuda) {
    Warn 'NVIDIA GPU detected - installing the CUDA build'
    & $LinPy -m pip install torch torchvision --quiet
} else {
    & $LinPy -m pip install torch torchvision --quiet --index-url https://download.pytorch.org/whl/cpu
}
& $LinPy -m pip install controlnet_aux --quiet
if ($LASTEXITCODE -ne 0) { Die 'Could not install the annotator dependencies.' }

Say 'Downloading the lineart model (~17 MB, once)'
$warm = @'
from controlnet_aux import LineartDetector
LineartDetector.from_pretrained("lllyasviel/Annotators")
print("    model ready")
'@
$warm | & $LinPy -
if ($LASTEXITCODE -ne 0) { Warn 'Model not pre-downloaded; it will fetch on first use instead.' }

# --------------------------------------------------------------- launcher --
# The launcher records where the annotator venv went, because lineart_backend
# is resolved relative to it and this path is not the Linux default.
$launcher = Join-Path $AppDir 'T-Tracer.cmd'
@"
@echo off
set "LINEART_PY=$LinPy"
start "" "$($MainVenv)\Scripts\pythonw.exe" "$AppDir\main.py" %*
"@ | Set-Content -Path $launcher -Encoding ASCII

if (-not $NoShortcuts) {
    Say 'Creating shortcuts'
    $ws = New-Object -ComObject WScript.Shell
    foreach ($dir in @([Environment]::GetFolderPath('Desktop'),
                       (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'))) {
        try {
            $lnk = $ws.CreateShortcut((Join-Path $dir 'T-Tracer.lnk'))
            $lnk.TargetPath       = $launcher
            $lnk.WorkingDirectory = $AppDir
            $lnk.Description      = 'Turn a customer logo into a laser-ready SVG'
            $lnk.IconLocation     = (Join-Path $AppDir 'static\icon.ico')
            $lnk.Save()
        } catch { Warn "Could not create a shortcut in $dir" }
    }
}

Write-Host ''
Write-Host '  Done.' -ForegroundColor Green
Write-Host '  Launch it from the Start Menu, or run T-Tracer.cmd here.'
Write-Host ''
