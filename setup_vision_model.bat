@echo off
echo [System] Starting Medical Vision Model Setup
echo ===================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Python is not installed or not in PATH
    echo Please install Python 3.8 or later from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [System] Step 1: Setting up Python Environment
powershell -ExecutionPolicy Bypass -File setup_environment.ps1
if %errorlevel% neq 0 (
    echo [Error] Failed to set up Python environment
    pause
    exit /b 1
)

echo.
echo [System] Step 2: Downloading Model
if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe Models/vision_model/model_downloader.py
) else (
    echo [Error] Virtual environment Python not found
    echo Please ensure Step 1 completed successfully
    pause
    exit /b 1
)

if %errorlevel% neq 0 (
    echo [Error] Failed to download model
    pause
    exit /b 1
)

echo.
echo [Success] Setup Complete!
pause