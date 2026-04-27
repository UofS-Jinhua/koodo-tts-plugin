@echo off

:: If launched by Koodo, bypass the minimized window creation
if "%LAUNCHED_BY_KOODO%" == "1" goto :skip_minimize

if not "%MINIMIZED%" == "1" (
    set MINIMIZED=1
    start /MIN cmd.exe /c "%~dpnx0"
    exit /b
)

:skip_minimize
title Audiobook TTS Server
echo =========================================
echo   Starting Local Audiobook TTS Server...
echo =========================================
echo.

:: Change to the directory where this batch file is located
cd /d "%~dp0"

:: Check if the virtual environment exists
if not exist ".venv-genie\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv-genie.
    echo Please make sure the setup was completed correctly.
    if not "%LAUNCHED_BY_KOODO%" == "1" pause
    exit /b 1
)

:: Activate the virtual environment
call .venv-genie\Scripts\activate.bat

:: Start the server using uvicorn
echo [INFO] Starting FastAPI server with Uvicorn...
python -m uvicorn app:app --host 127.0.0.1 --port 8000

:: If launched by Koodo, exit immediately so the CMD window closes
if "%LAUNCHED_BY_KOODO%" == "1" exit /b

echo.
pause
