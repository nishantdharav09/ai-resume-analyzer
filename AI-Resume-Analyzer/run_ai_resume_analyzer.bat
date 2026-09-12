@echo off
title AI Resume Analyzer
cd /d "%~dp0"

echo ==========================================
echo       AI Resume Analyzer
echo ==========================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Make sure this BAT file is inside the AI-Resume-Analyzer folder.
    echo.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"

echo Starting Streamlit app...
echo.

python -m streamlit run app.py

echo.
echo App stopped.
pause
