@echo off
title AI Resume Analyzer
cd /d "%~dp0"

echo ==========================================
echo        AI Resume Analyzer
echo ==========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not available in PATH.
    echo Please install Python 3 and try again.
    echo.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Installing required libraries...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo ERROR: Could not update pip.
    echo.
    pause
    exit /b 1
)

venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Could not install required libraries.
    echo.
    pause
    exit /b 1
)

echo.
echo Starting AI Resume Analyzer...
echo.

venv\Scripts\python.exe -m streamlit run app.py

echo.
echo ==========================================
echo        AI Resume Analyzer Stopped
echo ==========================================
pause
