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
    
    # FTSA uses adaptive batch sizing
    if config['training']['batch_size'] == 'auto':
        from .gpu_utils import get_gpu_info
        gpu_info = get_gpu_info()
        print(f"GPU detected: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")
        print(f"Auto-batch will be determined during training")
    
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
        "framework": "SmartAudit",
        "model_type": "student",
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
def check_readiness(eval_results: Dict, config: Dict):
    print("\n" + "="*50)
    print("Production Readiness Check")
    print("="*50)
    
    recall_vuln = eval_results.get('eval_recall_vulnerable', 0)
    f1_macro = eval_results.get('eval_f1_macro', 0)
    accuracy = eval_results.get('eval_accuracy', 0)
    
    # thresholds from config
    min_recall = config['metrics'].get('minimum_recall_vulnerable', 0.90)
    min_f1 = config['metrics'].get('minimum_f1_macro', 0.85)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f} (target: {min_f1}+)")
    print(f"  Recall (Vulnerable): {recall_vuln:.4f} (target: {min_recall}+)")
    
    is_ready = recall_vuln >= min_recall and f1_macro >= min_f1
    
    if is_ready:
        print("\nModel is ready for production!")
        print("  Meets all performance criteria")
    else:
        print("\nModel needs more training")
        if recall_vuln < min_recall:
            print(f"  - Improve vulnerable recall by {min_recall - recall_vuln:.4f}")
        if f1_macro < min_f1:
            print(f"  - Improve F1 macro by {min_f1 - f1_macro:.4f}")
    
    return is_ready