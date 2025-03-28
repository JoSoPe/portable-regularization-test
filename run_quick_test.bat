@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ========================================
echo    Running CUDA Availability Test
echo ========================================

REM Set project root to current directory
SET "PROJECT_ROOT=%~dp0"
SET "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"
SET "PIP_EXE=%PROJECT_ROOT%venv\Scripts\pip.exe"

REM Check if venv exists
IF NOT EXIST "%PYTHON_EXE%" (
    echo Creating virtual environment...
    python -m venv "%PROJECT_ROOT%venv"
)

REM Install PyTorch (CUDA 12.1 if supported)
echo Checking NVIDIA driver version...
FOR /F "tokens=3 delims= " %%A IN ('nvidia-smi ^| findstr /i "Driver Version"') DO (
    SET "DRIVER_VERSION=%%A"
    GOTO FoundDriver
)

:NoDriver
echo Could not detect NVIDIA driver version. Installing CPU-only PyTorch.
SET "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
GOTO InstallTorch

:FoundDriver
echo NVIDIA Driver Version Detected: %DRIVER_VERSION%
FOR /F "tokens=1 delims=." %%V IN ("%DRIVER_VERSION%") DO (
    SET "DRIVER_MAJOR=%%V"
)

IF %DRIVER_MAJOR% GEQ 530 (
    echo ✅ Your driver supports CUDA 12.1
    SET "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
) ELSE (
    echo ⚠️  Your driver may not support CUDA 12.1
    SET "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
)

:InstallTorch
echo Installing PyTorch from: %TORCH_INDEX%
"%PIP_EXE%" install --upgrade pip
"%PIP_EXE%" install torch torchvision torchaudio --index-url %TORCH_INDEX%

REM Run quickTest.py using venv Python directly
echo Running quickTest.py...
"%PYTHON_EXE%" "%PROJECT_ROOT%quickTest.py"

echo ========================================
echo        Test Complete
echo ========================================
pause
