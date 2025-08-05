#!/bin/bash

echo "Cloud Training Setup"
echo "===================="

# Find Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python not found"
    exit 1
fi

echo "Python: $($PYTHON_CMD --version)"

# Check GPU
echo -e "\nGPU Information:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "No GPU detected"
fi

# Virtual environment (optional)
if [ "$USE_VENV" = "true" ]; then
    echo -e "\nCreating virtual environment..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo -e "\nInstalling dependencies..."
$PYTHON_CMD -m pip install --upgrade pip

# Fix blinker package issue (distutils installed)
echo "Fixing blinker package..."
$PYTHON_CMD -m pip install --ignore-installed blinker 2>/dev/null || true

# Install requirements
echo "Installing requirements..."
$PYTHON_CMD -m pip install -r requirements.txt --exists-action i

# Extract data if available
if [ -f "processed_data.zip" ]; then
    echo -e "\nExtracting data..."
    unzip -q processed_data.zip
    echo "Data extracted"
else
    echo "Warning: processed_data.zip not found"
fi

# Create directories
mkdir -p output cache logs src/

# Environment setup
if [ -f ".env" ]; then
    echo -e "\nLoading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Test MLflow connection
if [ ! -z "$MLFLOW_TRACKING_URI" ]; then
    echo -e "\nTesting MLflow..."
    $PYTHON_CMD -c "
import requests
try:
    r = requests.get('$MLFLOW_TRACKING_URI/health', timeout=5)
    print('MLflow: OK' if r.status_code == 200 else f'MLflow: Error {r.status_code}')
except:
    print('MLflow: Connection failed')
"
fi

# HuggingFace login
if [ ! -z "$HF_TOKEN" ]; then
    echo -e "\nConfiguring HuggingFace..."
    $PYTHON_CMD -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
fi

# Set CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

echo -e "\nSetup complete!"
echo ""
echo "To start training: $PYTHON_CMD train.py"
echo "To test GPU: $PYTHON_CMD scripts/test_gpu.py"
echo "To monitor: watch -n 1 nvidia-smi"