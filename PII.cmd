@echo off
rem ==========================================================================
rem  PII.cmd - PII-AGENT v002 interactive launcher (persistent terminal menu).
rem  Mode select: [F]lash (bin\pii-agent-flash.exe) or [D]eep (bin\pii-agent.exe)
rem  Main menu loops until 0 (Exit); window never closes automatically.
rem ==========================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 ( echo Cannot enter %~dp0 & pause & exit /b 1 )

:mode_select
cls
echo PII-AGENT v002 - choose operating mode:
echo   [F]lash  inference only (bin\pii-agent-flash.exe)
echo   [D]eep   full engine    (bin\pii-agent.exe)
echo.
set "MODE="
set /p MODE="Mode [F/D]? "
if /i "%MODE%"=="F" ( set "EXE=bin\pii-agent-flash.exe" & set "MODENAME=FLASH" & goto :menu )
if /i "%MODE%"=="FLASH" ( set "EXE=bin\pii-agent-flash.exe" & set "MODENAME=FLASH" & goto :menu )
if /i "%MODE%"=="D" ( set "EXE=bin\pii-agent.exe" & set "MODENAME=DEEP" & goto :menu )
if /i "%MODE%"=="DEEP" ( set "EXE=bin\pii-agent.exe" & set "MODENAME=DEEP" & goto :menu )
echo Invalid choice. Type F or D.
pause
goto :mode_select

:menu
cls
echo PII-AGENT v002 - Main Menu  [Mode: %MODENAME%]
echo ==================================================
echo   1. Selftest
echo   2. Mask
echo   3. Unmask
echo   4. Purge
if /i "%MODENAME%"=="DEEP" echo   5. Train
echo   0. Exit
echo ==================================================
set "CHOICE="
set /p CHOICE="Select option: "
if "%CHOICE%"=="1" goto :do_selftest
if "%CHOICE%"=="2" goto :do_mask
if "%CHOICE%"=="3" goto :do_unmask
if "%CHOICE%"=="4" goto :do_purge
if /i "%MODENAME%"=="DEEP" if "%CHOICE%"=="5" goto :do_train
if "%CHOICE%"=="0" goto :do_exit
echo Invalid choice.
pause
goto :menu

:check_exe
if not exist "%EXE%" ( echo Missing %EXE% - build first with build.cmd. & pause & exit /b 1 )
exit /b 0

:do_selftest
call :check_exe || goto :menu
"%EXE%" selftest
pause
goto :menu

:do_mask
call :check_exe || goto :menu
echo Drag and drop the file path, or type it.
set "FILE="
set /p FILE="File path: "
set "FILE=%FILE:"=%"
if "%FILE%"=="" ( echo No file given. & pause & goto :menu )
"%EXE%" mask "%FILE%"
pause
goto :menu

:do_unmask
call :check_exe || goto :menu
set "UUID="
set /p UUID="UUID (or masked PDF path): "
set "UUID=%UUID:"=%"
if "%UUID%"=="" ( echo No UUID given. & pause & goto :menu )
"%EXE%" unmask "%UUID%"
pause
goto :menu

:do_purge
call :check_exe || goto :menu
set "PUUID="
set /p PUUID="UUID: "
set "PUUID=%PUUID:"=%"
if "%PUUID%"=="" ( echo No UUID given. & pause & goto :menu )
"%EXE%" purge "%PUUID%"
pause
goto :menu

:do_train
if /i not "%MODENAME%"=="DEEP" ( echo Train needs Deep mode. & pause & goto :menu )
call :check_exe || goto :menu
echo Drag and drop the file-or-folder path, or type it.
set "TPATH="
set /p TPATH="Train path: "
set "TPATH=%TPATH:"=%"
if "%TPATH%"=="" ( echo No path given. & pause & goto :menu )
set "EPOCHS="
set /p EPOCHS="Epochs (number, 0/inf/empty = infinite): "
set "EPOCHS=%EPOCHS:"=%"
if "%EPOCHS%"=="" (
  "%EXE%" train "%TPATH%"
) else (
  "%EXE%" train "%TPATH%" "%EPOCHS%"
)
pause
goto :menu

:do_exit
exit /b 0
