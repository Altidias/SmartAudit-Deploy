#!/bin/bash

echo "Student Model Training Setup"
echo "========================================"

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

echo -e "\nGPU Information:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "ERROR: No GPU detected. requires GPU for training."
    exit 1
fi

if [ "$USE_VENV" = "true" ] || [ -z "$USE_VENV" ]; then
    echo -e "\nCreating venv..."
    $PYTHON_CMD -m venv smartaudit_env
    source smartaudit_env/bin/activate
fi

echo -e "\nInstalling dependencies..."
$PYTHON_CMD -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
$PYTHON_CMD -m pip install -r requirements.txt --upgrade

if [ -f "processed_data.zip" ]; then
    echo -e "\nExtracting dataset..."
    unzip -q processed_data.zip
    echo "Dataset extracted"
    
    # Verify vulnerability types file
    if [ -f "processed_data/vuln_types.json" ]; then
        NUM_TYPES=$(python -c "import json; print(len(json.load(open('processed_data/vuln_types.json'))))")
        echo "Vulnerability types loaded: $NUM_TYPES"
    fi
else
    echo "ERROR: dataset (processed_data.zip) not found!"
    exit 1
fi

mkdir -p output cache logs src/

if [ -f ".env" ]; then
    echo -e "\nLoading environment variables..."
    export $(grep -v '^#' .env | xargs)
fi

if [ ! -z "$MLFLOW_TRACKING_URI" ]; then
    echo -e "\nTesting MLflow connection..."
    $PYTHON_CMD -c "
import requests
try:
    r = requests.get('$MLFLOW_TRACKING_URI/health', timeout=5)
    print('MLflow: Connected' if r.status_code == 200 else f'MLflow: Error {r.status_code}')
except:
    print('MLflow: Not available (training will continue without MLflow)')
"
fi

# HuggingFace login (required for model download)
if [ ! -z "$HF_TOKEN" ]; then
    echo -e "\nConfiguring HuggingFace access..."
    $PYTHON_CMD -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
else
    echo -e "\nWARNING: HF_TOKEN not set. You may need to login manually."
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

echo -e "\n========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Model: Qwen2.5-Coder-7B-Instruct (Student)"
echo "Framework: FTSmartAudit Knowledge Distillation"
echo ""
echo "To start training: $PYTHON_CMD train.py"
echo "To test GPU: $PYTHON_CMD scripts/test_gpu.py"
echo "To evaluate model: $PYTHON_CMD scripts/evaluate.py"
echo "To monitor GPU: watch -n 1 nvidia-smi"
echo ""