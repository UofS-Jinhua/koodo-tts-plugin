@echo off
setlocal

:: NOTE: keep this file ASCII-only. cmd.exe mis-parses batch files that carry a
:: lot of multi-byte text (it seeks by byte offset), which silently corrupts
:: lines further down. Verified: ~80 non-ASCII chars is fine, ~370 breaks it.

:: Skip the minimize dance when Koodo launched us (the plugin wants a real window)
if not "%LAUNCHED_BY_KOODO%" == "1" (
    if not "%MINIMIZED%" == "1" (
        set MINIMIZED=1
        start /MIN cmd.exe /c "%~dpnx0"
        exit /b
    )
)

:: The Koodo plugin kills this window by title on exit. Renaming it here means
:: renaming it in koodo_plugin/neurolink_tts_plugin.json too.
title Neurolink TTS Server
echo =========================================
echo   Starting Neurolink TTS Server...
echo =========================================
echo.

:: --- Target directory ---------------------------------------------------
:: This path MUST stay hardcoded to TTS_Server. Do not "clean it up" to %~dp0:
:: %~dp0 is the audiobook folder, not this one.
:: Reason: server.py locates GPT-SoVITS relative to the current directory --
::     GPT_SOVITS_PATH = os.path.join(os.getcwd(), "GPT-SoVITS")
:: so the process CWD must be exactly TTS_Server. One level off and the model
:: fails to load. If the repo moves, edit this single line.
set "TTS_SERVER_DIR=C:\Codes\Neurolink-Node\TTS_Server"

if not exist "%TTS_SERVER_DIR%\" (
    echo [ERROR] TTS Server directory not found:
    echo         %TTS_SERVER_DIR%
    echo         Edit TTS_SERVER_DIR in this script if the repo has moved.
    goto :fail
)

cd /d "%TTS_SERVER_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to change directory to %TTS_SERVER_DIR%
    goto :fail
)

:: --- Environment checks --------------------------------------------------
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at %TTS_SERVER_DIR%\.venv
    echo         Please make sure the setup was completed correctly.
    goto :fail
)

if not exist "server.py" (
    echo [ERROR] server.py not found in %TTS_SERVER_DIR%
    goto :fail
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    goto :fail
)

:: --- Launch ---------------------------------------------------------------
echo [INFO] Working directory: %CD%
echo [INFO] Starting TTS Server (server.py)...
echo [INFO] Model loading can take over a minute on a cold start.
echo.

python server.py
set "EXITCODE=%ERRORLEVEL%"

:: Always stop on a crash, even under Koodo. Otherwise the window vanishes
:: instantly and the traceback is lost -- from the plugin side that looks
:: like "retried a few times, then no audio", which is miserable to debug.
if not "%EXITCODE%" == "0" (
    echo.
    echo =========================================
    echo [ERROR] Server exited with code %EXITCODE%.
    echo         The error output above is the reason.
    echo =========================================
    pause
    exit /b %EXITCODE%
)

if "%LAUNCHED_BY_KOODO%" == "1" exit /b 0

echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
