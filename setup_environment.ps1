# PowerShell script to set up Python environment for MFDIS

Write-Host "`n[System] Setting up Python Environment for Medical Vision Model" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "`n[Info] Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "`n[Error] Python not found! Please install Python 3.8 or later" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment if it doesn't exist
$venvPath = ".\venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "`n[System] Creating virtual environment..." -ForegroundColor Yellow
    try {
        python -m venv $venvPath
        Write-Host "[Success] Virtual environment created successfully" -ForegroundColor Green
    } catch {
        Write-Host "[Error] Failed to create virtual environment" -ForegroundColor Red
        Write-Host $_.Exception.Message
        exit 1
    }
} else {
    Write-Host "`n[Info] Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n[System] Activating virtual environment..." -ForegroundColor Yellow
try {
    & "$venvPath\Scripts\Activate.ps1"
    Write-Host "[Success] Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "[Error] Failed to activate virtual environment" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Install required packages
Write-Host "`n[System] Installing required packages..." -ForegroundColor Yellow

# Update pip first
Write-Host "`n[System] Updating pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA
Write-Host "`n[System] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
$packages = @(
    "transformers>=4.40.0",
    "accelerate>=0.21.0",
    "safetensors>=0.3.1",
    "bitsandbytes>=0.41.0",
    "pillow>=9.5.0",
    "huggingface_hub>=0.16.0",
    "psutil",
    "GPUtil"
)

foreach ($package in $packages) {
    Write-Host "`n[System] Installing $package..." -ForegroundColor Yellow
    try {
        python -m pip install $package
        Write-Host "[Success] Installed $package" -ForegroundColor Green
    } catch {
        Write-Host "[Warning] Failed to install $package" -ForegroundColor Yellow
        Write-Host $_.Exception.Message
    }
}

# Verify installation
Write-Host "`n[System] Verifying installations..." -ForegroundColor Green

$verificationScript = @'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
'@

python -c $verificationScript

Write-Host "`n[Success] Environment setup complete!" -ForegroundColor Green
Write-Host "[Info] To activate this environment in the future, run:" -ForegroundColor Yellow
Write-Host "       .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan