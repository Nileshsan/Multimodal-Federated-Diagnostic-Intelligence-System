#!/bin/bash

echo "🚀 Installing requirements for Llama 3.2 11B Vision Model..."
echo "================================================"

# Update pip first
echo "📦 Updating pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core transformers libraries
echo "🤗 Installing Transformers ecosystem..."
pip install transformers>=4.40.0
pip install accelerate>=0.21.0
pip install safetensors>=0.3.1

# Quantization support
echo "⚡ Installing quantization libraries..."
pip install bitsandbytes>=0.41.0

# Vision processing
echo "👁️ Installing vision libraries..."
pip install pillow>=9.5.0

# Hugging Face Hub for downloading
echo "🌐 Installing Hugging Face Hub..."
pip install huggingface_hub>=0.16.0

# System monitoring utilities
echo "📊 Installing system monitoring..."
pip install psutil GPUtil

echo ""
echo "✅ Installation completed!"
echo "🔍 Verifying installation..."

# Verify CUDA
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA not available - model will run on CPU (very slow)')
"

# Verify Transformers
python -c "
import transformers
print(f'Transformers version: {transformers.__version__}')
"