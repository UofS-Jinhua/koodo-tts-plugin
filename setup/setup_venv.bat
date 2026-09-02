@echo off
setlocal

:: NOTE: keep this file ASCII-only. cmd.exe seeks batch files by byte offset and
:: mis-parses ones carrying a lot of multi-byte text, silently corrupting lines
:: further down. Verified: ~80 non-ASCII chars is fine, ~370 breaks it.

title Audiobook TTS - Environment Setup
echo =========================================
echo   Audiobook TTS environment setup
echo =========================================
echo.

:: This script lives in setup\, the project root is one level up
cd /d "%~dp0.."

:: --- 1. Locate a usable Python -------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3.10+ and add it to PATH.
    goto :fail
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required. Found:
    python --version
    goto :fail
)

:: --- 2. Create the virtual environment ------------------------------------
if exist ".venv-genie\Scripts\activate.bat" (
    echo [INFO] .venv-genie already exists, skipping creation.
) else (
    echo [INFO] Creating virtual environment .venv-genie ...
    python -m venv .venv-genie
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        goto :fail
    )
)

call .venv-genie\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate the virtual environment.
    goto :fail
)

python -m pip install --upgrade pip

:: --- 3. jieba_fast, with a fallback ---------------------------------------
:: genie-tts imports jieba_fast for Chinese G2P. It needs a C++ compiler to
:: build; when that is missing we install plain jieba and inject a shim module
:: named jieba_fast that redirects to it.
echo.
echo [INFO] Installing jieba_fast ...
python -m pip install jieba_fast==0.53
if errorlevel 1 (
    echo.
    echo [WARN] jieba_fast failed to build ^(usually missing MSVC 14.0+^).
    echo [WARN] Falling back to pure-Python jieba plus a jieba_fast shim.
    python -m pip install jieba
    if errorlevel 1 (
        echo [ERROR] jieba failed to install too. Check your network.
        goto :fail
    )
    python create_jieba_shim.py
    if errorlevel 1 (
        echo [ERROR] Failed to create the jieba_fast shim.
        goto :fail
    )
)

:: --- 4. Remaining dependencies --------------------------------------------
echo.
echo [INFO] Installing the remaining dependencies ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    goto :fail
)

:: --- 5. Self check ---------------------------------------------------------
echo.
echo [INFO] Verifying the installation ...
:: Do NOT `import genie_tts` here. Importing it executes Core/Resources.py, which
:: at this point (models are downloaded in the NEXT step) prints a non-ASCII
:: warning and then waits on input(). On a GBK console that print raises
:: UnicodeEncodeError, so this check used to fail on a perfectly good install.
:: find_spec only locates the package without running any of its code.
:: nltk is needed by the English G2P, i.e. by the mixed Chinese/English reading.
python -c "import fastapi, uvicorn, soundfile, onnxruntime, nltk, jieba_fast; import sys; from importlib.util import find_spec; sys.exit(0 if find_spec('genie_tts') else 1)"
if errorlevel 1 (
    echo [ERROR] Self check failed, some dependencies are missing.
    goto :fail
)
echo   all core imports OK

echo.
echo =========================================
echo   Setup complete.
echo.
echo   Next: download the models
echo     .venv-genie\Scripts\python.exe setup\download_models.py
echo   Then double-click start_server.bat to launch the service.
echo =========================================
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
