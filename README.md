# Smart Audit Vulnerability Detection Training

Fine-tuning LLMs to detect vulnerabilities in smart contracts.

## Setup

1. Create environment and install dependencies:
```bash
./setup.sh
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your MLflow URI and HuggingFace token
```

3. Test GPU memory (optional):
```bash
python scripts/test_gpu.py
```

## Training

Start training with:
```bash
python train.py
```

The script will:
- Automatically detect optimal batch size for your GPU
- Resume from checkpoints if interrupted
- Track experiments with MLflow (if configured)
- Save models and LoRA adapters separately

## Configuration

Edit `config.yaml` to adjust:
- Model selection
- Training hyperparameters
- LoRA configuration
- MLflow tracking

## Project Structure

```
src/
├── config.py       # Configuration management
├── data.py         # Dataset loading
├── gpu_utils.py    # GPU memory management
├── metrics.py      # Evaluation metrics
├── mlflow_utils.py # Experiment tracking
├── model.py        # Model and LoRA setup
└── trainer.py      # Custom trainer with weighted loss

scripts/
├── test_gpu.py     # GPU memory testing
└── evaluate.py     # Model evaluation
```
