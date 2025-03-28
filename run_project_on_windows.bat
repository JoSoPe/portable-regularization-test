@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ========================================
echo   Gradient Descent Project Runner
echo ========================================

REM Set project root to current directory
SET PROJECT_ROOT=%~dp0

REM Step 1: Create virtual environment
IF NOT EXIST "%PROJECT_ROOT%venv" (
    echo Creating virtual environment...
    python -m venv "%PROJECT_ROOT%venv"
)

REM Step 2: Activate virtual environment
echo Activating virtual environment...
call "%PROJECT_ROOT%venv\Scripts\activate.bat"

REM Step 3: Upgrade pip and install requirements
echo Installing requirements...
pip install --upgrade pip
pip install -r "%PROJECT_ROOT%requirements.txt"

REM Step 4: Run the main Python script
echo Running main.py...
python "%PROJECT_ROOT%main.py"

echo ========================================
echo       Script execution complete
echo ========================================
pause
