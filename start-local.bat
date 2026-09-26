@echo off
setlocal EnableExtensions DisableDelayedExpansion
title STL simulator
pushd "%~dp0" >nul 2>nul
if errorlevel 1 goto folder_error
set "STUDIO_LOG=%CD%\studio-startup.log"
if not exist "scripts\bootstrap-windows.ps1" goto missing_files
echo STL simulator
echo Python will be set up automatically if needed. Keep this window open.
echo.
rem ExecutionPolicy applies to this child process only; no machine or user policy is changed.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\bootstrap-windows.ps1" %*
set "STUDIO_EXIT=%ERRORLEVEL%"
if not "%STUDIO_EXIT%"=="0" goto failed
popd
exit /b 0

:missing_files
> "%STUDIO_LOG%" echo Required setup files are missing. Extract the ENTIRE STL-simulator.zip before starting.
>> "%STUDIO_LOG%" echo start-local.bat must be beside launch.py, scripts, server, and web.
set "STUDIO_EXIT=1"
goto failed

:failed
echo.
echo STARTUP FAILED. The window will stay open so you can read the error.
echo Full startup log: "%STUDIO_LOG%"
if exist "%STUDIO_LOG%" start "" notepad.exe "%STUDIO_LOG%"
echo Press any key to close. Please share the final error lines if setup fails.
pause >nul
popd
exit /b %STUDIO_EXIT%

:folder_error
echo Cannot open the application folder. Extract the entire ZIP before running.
pause
exit /b 1
