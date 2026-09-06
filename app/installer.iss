; T-Tracer - Inno Setup script.
;
; Produces T-Tracer-Setup.exe: a small BOOTSTRAPPER, roughly 2 MB.
;
; Why a bootstrapper rather than a self-contained build. PyTorch alone is about
; 2 GB, so a bundled installer lands somewhere north of 2.5 GB - past Discord's
; 25 MB attachment limit by a factor of a hundred, and past GitHub's 2 GB
; release-asset limit outright. This ships the app code and lets install.ps1
; fetch Python and the heavy dependencies on first run. The download happens
; once, on the user's machine, from python.org and PyPI.
;
; Built by .github/workflows/build-installer.yml on a Windows runner. It cannot
; be built or tested on Linux.

#define AppName        "T-Tracer"
#define AppPublisher   "Tyler Pfister"
#define AppExeName     "T-Tracer.cmd"
#ifndef AppVersion
  #define AppVersion   "0.1.0"
#endif

[Setup]
AppId={{8F3C1A94-6B2E-4D7A-9E51-2C4B8D6F1A03}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\T-Tracer
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=T-Tracer-Setup
OutputDir=..\dist
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-user by default so no admin prompt is needed - the point is that someone
; can install this on a work machine they do not administer.
PrivilegesRequiredOverridesAllowed=dialog
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\app\static\icon.ico
LicenseFile=..\LICENSE
SetupIconFile=static\icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
; The app and the tracing pipeline. No venvs, no models, no corpus - install.ps1
; builds all of that on the target machine.
Source: "..\app\*";           DestDir: "{app}\app";      Flags: ignoreversion recursesubdirs; \
    Excludes: "*.pyc,__pycache__\*,.work\*,.chrome-profile\*,T-Tracer,T-Tracer.cmd"
Source: "..\_scripts\*.py";   DestDir: "{app}\_scripts"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}";         Flags: ignoreversion
Source: "..\_shared\laser-output-rules.md"; DestDir: "{app}\_shared"; Flags: ignoreversion skipifsourcedoesntexist

[Run]
; The long pole. Shown in a window rather than hidden, because it downloads
; ~2 GB and a silent installer that appears frozen for ten minutes gets killed.
Filename: "powershell.exe"; \
    Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\app\install.ps1"" -AppDir ""{app}\app"" -NoShortcuts"; \
    StatusMsg: "Installing Python and dependencies (this downloads ~2 GB and takes a while)..."; \
    Flags: waituntilterminated

[Icons]
Name: "{group}\{#AppName}";           Filename: "{app}\app\{#AppExeName}"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\static\icon.ico"
Name: "{autodesktop}\{#AppName}";     Filename: "{app}\app\{#AppExeName}"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\static\icon.ico"; Tasks: desktopicon

[UninstallDelete]
; Everything install.ps1 created outside {app}. Without this, uninstalling
; leaves a couple of GB of venv behind and nobody ever finds it.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{localappdata}\t-tracer"

[Messages]
WelcomeLabel2=This installs [name/ver].%n%nIt will download Python and PyTorch if they are not already present - about 2 GB on first install. Later updates reuse them.
