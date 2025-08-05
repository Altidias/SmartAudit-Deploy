#!/bin/bash

echo "=========================================="
echo "Cloud Training Setup"
echo "=========================================="
echo ""

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if command_exists python3; then
    PYTHON_CMD=python3
elif command_exists python; then
    PYTHON_CMD=python
else
    echo "Python not found!"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

echo -e "\nGPU Information:"
if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "nvidia-smi not found - GPU might not be available"
fi

if [ "$USE_VENV" = "true" ]; then
    echo -e "\nCreating virtual environment..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
fi

echo -e "\nInstalling dependencies..."
$PYTHON_CMD -m pip install --upgrade pip

$PYTHON_CMD -m pip install --ignore-installed blinker 2>/dev/null || true

echo "Installing requirements..."
$PYTHON_CMD -m pip install -r requirements.txt --exists-action i

if ! $PYTHON_CMD -c "import sklearn" 2>/dev/null; then
    echo "scikit-learn not found, installing..."
    $PYTHON_CMD -m pip install scikit-learn
fi

if [ -f "processed_data.zip" ]; then
    echo -e "\nExtracting processed data..."
    unzip -q processed_data.zip
    echo "Data extracted successfully"
else
    echo "processed_data.zip not found - you'll need to process data manually"
fi

echo -e "\nCreating directories..."
mkdir -p output
mkdir -p cache
mkdir -p logs

if [ -f ".env" ]; then
    echo -e "\nLoading environment variables..."
    export $(grep -v '^#' .env | xargs)
fi

if [ ! -z "$MLFLOW_TRACKING_URI" ]; then
    echo -e "\nTesting MLflow connection..."
    $PYTHON_CMD -c "from utils import test_mlflow_connection; test_mlflow_connection('$MLFLOW_TRACKING_URI')"
fi

if [ ! -z "$HF_TOKEN" ]; then
    echo -e "\nConfiguring HuggingFace..."
    $PYTHON_CMD -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

echo -e "\n=========================================="
echo "Setup complete!"
echo ""
echo "To start training:"
echo "  $PYTHON_CMD train_cloud.py"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo "=========================================="