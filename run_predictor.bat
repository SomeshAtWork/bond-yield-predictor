@echo off
REM ============================================================================
REM  BOND YIELD PREDICTOR  -  LAUNCHER
REM ============================================================================
REM  Invoked by the RUN PREDICTION hyperlink in bond_predictor_launcher.xlsx.
REM  Also works as a direct double-click launcher.
REM
REM  Pipeline:
REM    - If launcher_bridge.py is present, it reads the current weights +
REM      horizon from bond_predictor_launcher.xlsx, writes predictor_config.json,
REM      and runs the predictor in non-interactive mode.
REM    - If launcher_bridge.py is missing (direct-run), falls back to
REM      interactive mode.
REM  Requires Anaconda installed at one of the usual paths (or python on PATH).
REM ============================================================================

setlocal

title Bond Yield Predictor v3.5

REM --- Find Anaconda python first, fall back to python on PATH ---
set PY_EXE=
if exist "%USERPROFILE%\anaconda3\python.exe"    set PY_EXE=%USERPROFILE%\anaconda3\python.exe
if exist "%USERPROFILE%\miniconda3\python.exe"   set PY_EXE=%USERPROFILE%\miniconda3\python.exe
if exist "C:\ProgramData\anaconda3\python.exe"   set PY_EXE=C:\ProgramData\anaconda3\python.exe
if exist "C:\ProgramData\miniconda3\python.exe"  set PY_EXE=C:\ProgramData\miniconda3\python.exe

if "%PY_EXE%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        set PY_EXE=python
    )
)

if "%PY_EXE%"=="" (
    echo ERROR: Python not found. Install Anaconda or add python to PATH.
    pause
    exit /b 1
)

REM --- Locate scripts (same dir as this .bat) ---
set SCRIPT_DIR=%~dp0
set PREDICTOR=%SCRIPT_DIR%bond_yield_predictor.py
set BRIDGE=%SCRIPT_DIR%launcher_bridge.py
set LAUNCHER=%SCRIPT_DIR%bond_predictor_launcher.xlsx

if not exist "%PREDICTOR%" (
    echo ERROR: bond_yield_predictor.py not found at:
    echo   %PREDICTOR%
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo   INDIAN GOVERNMENT BOND YIELD PREDICTOR  v3.5
echo ============================================================================
echo   Python  : %PY_EXE%
echo   Folder  : %SCRIPT_DIR%
echo ============================================================================
echo.

REM --- Decide which mode to run ---
if exist "%BRIDGE%" if exist "%LAUNCHER%" (
    echo   Mode: launcher-driven ^(bond_predictor_launcher.xlsx^)
    echo.
    "%PY_EXE%" "%BRIDGE%"
    set RC=%ERRORLEVEL%
    goto :after_run
)

echo   Mode: direct / interactive
echo.
"%PY_EXE%" "%PREDICTOR%"
set RC=%ERRORLEVEL%

:after_run
echo.
if %RC%==0 (
    echo   SUCCESS.
) else (
    echo   FAILED with exit code %RC%
    echo.
    echo   Common fixes:
    echo     - Make sure dataforpythontraining.xlsx exists in Downloads
    echo     - Close bond_predictions_output.xlsx if it is open
    echo     - Re-run from an Anaconda Prompt to see full tracebacks
)
echo.
pause
endlocal
