import yaml
import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

def load_config() -> Dict:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle auto batch size
    if config['training']['batch_size'] == 'auto':
        from .gpu_utils import get_gpu_info
        gpu_info = get_gpu_info()
        config['training']['batch_size'] = gpu_info['recommended_batch_size']
        config['training']['gradient_accumulation_steps'] = gpu_info['recommended_gradient_accumulation']
    
    # Handle auto run name
    if config['mlflow']['run_name'] == 'auto':
        from .mlflow_utils import create_run_name
        config['mlflow']['run_name'] = create_run_name()
    
    # Create directories
    for key, path in config['paths'].items():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return config

def save_training_info(config: Dict, output_dir: str):
    from .gpu_utils import get_gpu_info, get_system_info
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "gpu": get_gpu_info(),
        "system": get_system_info(),
        "environment": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "python": sys.version.split()[0]
        }
    }
    
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, item)))
            except:
                pass
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]
    return None

def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        sys.exit(1)

def check_production_readiness(eval_results: Dict, config: Dict):
    print("\n" + "="*50)
    print("PRODUCTION READINESS CHECK")
    print("="*50)
    
    recall_vuln = eval_results.get('eval_recall_vulnerable', 0)
    precision = eval_results.get('eval_precision', 0)
    min_recall = config['metrics']['minimum_recall']
    min_precision = config['metrics']['minimum_precision']
    
    if recall_vuln >= min_recall and precision >= min_precision:
        print("✓ Model is ready for production!")
        print(f"  Recall: {recall_vuln:.4f} (target: {min_recall}+)")
        print(f"  Precision: {precision:.4f} (target: {min_precision}+)")
    elif recall_vuln >= min_recall:
        print("⚠ Good recall but low precision - suitable as pre-filter")
        print(f"  Recall: {recall_vuln:.4f} ✓")
        print(f"  Precision: {precision:.4f} (target: {min_precision}+)")
    else:
        print("✗ Model needs more training")
        print(f"  Recall: {recall_vuln:.4f} (target: {min_recall}+)")
        print(f"  Precision: {precision:.4f} (target: {min_precision}+)")