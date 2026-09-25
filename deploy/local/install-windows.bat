@echo off
rem STL Simulator installer for Windows: double-click this file (after extracting the whole ZIP).
rem It starts installer-files\install-windows.ps1 with Windows PowerShell (execution policy bypassed for this run).
rem Options are passed through, e.g.  install-windows.bat -Port 8005 -InstallDir D:\STL-Simulator
setlocal
if not exist "%~dp0installer-files\install-windows.ps1" goto notextracted
if not exist "%~dp0installer-files\stl-pipe.cmd" goto notextracted
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer-files\install-windows.ps1" install -Pause %*
set "rc=%errorlevel%"
rem 0 = done, 20 = an error the installer already explained (it waited for Enter itself)
if "%rc%"=="0" exit /b 0
if "%rc%"=="20" exit /b 20
echo.
echo  [STL Simulator] PowerShell could not run the installer (exit code %rc%).
echo  Possible causes: a company/school policy blocks PowerShell scripts, or antivirus software
echo  blocked installer-files\install-windows.ps1. Please send a photo of this window to the lab.
echo.
pause
exit /b %rc%

:notextracted
echo.
echo  [STL Simulator] The installer files were not found next to this file.
echo  Please extract the whole ZIP first (right-click the ZIP - "Extract All..."),
echo  then run install-windows.bat inside the extracted folder.
echo.
pause
exit /b 1
