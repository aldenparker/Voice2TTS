; Inno Setup script for Voice2TTS.
;
; Built by build.ps1 after PyInstaller produces dist\Voice2TTS.
; Compile manually with:  iscc installer\Voice2TTS.iss
;
; Installs per-user by default (PrivilegesRequired=lowest) so no admin prompt is
; needed for the app itself. The only thing that genuinely needs elevation is the
; VB-CABLE driver, and that is handled at runtime by the setup wizard, where the
; user can see exactly what is being installed and why.

#define AppName "Voice2TTS"
#define AppPublisher "Voice2TTS"
#define AppURL "https://github.com/"
#define AppExeName "Voice2TTS.exe"

#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif

#ifndef SourceDir
  #define SourceDir "..\dist\Voice2TTS"
#endif

[Setup]
AppId={{7B3C9F1E-4A2D-4E7B-9C51-2F8E6D4A1B93}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE.txt
InfoBeforeFile=..\installer\BEFORE.txt
OutputDir=..\dist\installer
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
SetupIconFile=voice2tts.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ExtraDiskSpaceRequired=0
MinVersion=10.0
; Upgrades reuse the previous location instead of reverting to the default, so a
; user who installed elsewhere is not silently given a second copy.
UsePreviousAppDir=yes
UsePreviousGroup=yes
UsePreviousTasks=yes
; The app is a tray program and will usually be running during an update. Restart
; Manager closes it so its files can be replaced; without this a silent update
; fails on locked DLLs. RestartApplications=no because we relaunch ourselves via
; the /relaunch flag below, which avoids a double launch.
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; \
    GroupDescription: "Shortcuts:"
Name: "startupicon"; Description: "Start {#AppName} when I sign in"; \
    GroupDescription: "Startup:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs
; The GPL requires the licence to accompany the distributed work, so these are
; installed alongside the app rather than only shown during setup.
Source: "..\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\COPYING"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
; The console build, not the windowed one -- a windowed bootloader has no stdout,
; so --cli through {#AppExeName} would print nothing.
Name: "{group}\{#AppName} (console)"; Filename: "{app}\Voice2TTS-console.exe"; \
    Parameters: "--cli"; Comment: "Run with a visible log window"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    Tasks: startupicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName} and run setup"; \
    Flags: nowait postinstall skipifsilent
; Silent in-app updates pass /relaunch=1. The entry above is skipifsilent, so
; without this one an auto-update would leave the app closed.
Filename: "{app}\{#AppExeName}"; Flags: nowait; Check: ShouldRelaunch

[UninstallDelete]
; Generated at runtime, so Inno does not track them.
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
// True when launched by the in-app updater, which passes /relaunch=1.
function ShouldRelaunch: Boolean;
begin
  Result := ExpandConstant('{param:relaunch|0}') = '1';
end;

// Offer to clean up the downloaded GPU pack and models, which live outside {app}
// and can be well over a gigabyte. Config is left alone unless asked for.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  CacheDir, ConfigDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    CacheDir := ExpandConstant('{localappdata}\Voice2TTS');
    ConfigDir := ExpandConstant('{userappdata}\Voice2TTS');

    if DirExists(CacheDir) then
    begin
      if MsgBox('Also delete downloaded models and GPU libraries?' #13#10 #13#10
                + CacheDir + #13#10 #13#10
                + 'This can free over a gigabyte. Choose No to keep them for a '
                + 'future reinstall.',
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(CacheDir, True, True, True);
    end;

    if DirExists(ConfigDir) then
    begin
      if MsgBox('Also delete your settings and downloaded voices?' #13#10 #13#10
                + ConfigDir,
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(ConfigDir, True, True, True);
    end;
  end;
end;
