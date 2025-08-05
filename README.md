# Cloud Training Deployment

This package contains everything needed to train a smart contract vulnerability detection model on cloud GPUs.

## Contents

- `train_cloud.py` - Main training script with auto-GPU detection
- `config.yaml` - Training configuration (auto-adjusts to GPU)
- `setup.sh` - One-click environment setup
- `utils.py` - Helper utilities
- `requirements.txt` - Python dependencies
- `processed_data.zip` - Your preprocessed training data


## Local mlflow setup

1. install mlflow:
   ```bash
   pip install mlflow
   ```

2. create data directory:
   ```bash
   mkdir C:\mlflow_data
   ```

3. start mlflow server:
   ```bash
   mlflow server --backend-store-uri file:///C:/mlflow_data/mlruns --default-artifact-root file:///C:/mlflow_data/artifacts --host 0.0.0.0 --port 5000
   ```

4. make accessible from cloud using ngrok or similar service, or just host but make sure to use auth:
   - option a: use ngrok
     ```bash
     ngrok http 5000
     ```
   - option b: use tailscale for secure connection

## Cloud GPU setup

1. upload training package

2. setup environment:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   (install unzip and other necassary progs if needed)

3. configure mlflow uri:
   ```bash
   # edit .env file
   MLFLOW_TRACKING_URI=https://your-ngrok-url.ngrok.io
   ```

4. start training:
   ```bash
   python train_cloud.py
   ```

## monitoring

- **MLflow ui**: http://localhost:5000
- **GPU usage**: `watch -n 1 nvidia-smi`
- **Training Logs**: check output directory

## Key metrics

- **eval_f1**: overall performance
- **eval_recall_vulnerable**: catching vulnerabilities
- **eval_precision**: accuracy of alerts
