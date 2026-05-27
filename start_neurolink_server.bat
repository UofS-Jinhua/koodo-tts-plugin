@echo off

:: 检查是否是通过 Koodo 插件调用的（避免出现不需要的命令行窗口和脱机问题）
if not "%LAUNCHED_BY_KOODO%" == "1" (
    :: 如果是手动双击启动，则最小化运行
    if not "%MINIMIZED%" == "1" (
        set MINIMIZED=1
        start /MIN cmd.exe /c "%~dpnx0"
        exit /b
    )
)

title Neurolink TTS Server
echo =========================================
echo   Starting Neurolink TTS Server...
echo =========================================
echo.

:: 切换到目标 TTS Server 目录
cd /d "C:\Codes\Neurolink-Node\TTS_Server"

:: 检查并激活虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv.
    echo Please make sure the setup was completed correctly.
    if not "%LAUNCHED_BY_KOODO%" == "1" pause
    exit /b 1
)

call .venv\Scripts\activate.bat

:: 启动 server.py
echo [INFO] Starting TTS Server (server.py)...
python server.py

:: 如果是 Koodo 调用的，就不需要暂停
if "%LAUNCHED_BY_KOODO%" == "1" exit /b

echo.
pause