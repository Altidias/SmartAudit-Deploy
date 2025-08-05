import mlflow
import mlflow.pytorch
import requests
import os
from datetime import datetime
from transformers import TrainerCallback
from typing import Dict, Optional

def create_run_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "-").lower()[:20]
    else:
        gpu_name = "cpu"
    
    return f"student_{gpu_name}_{timestamp}"

def setup_mlflow(config: Dict) -> Optional[str]:
    """Setup MLflow for experiment tracking"""
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
    
    print(f"\nSetting up MLflow tracking...")
    
    # Test con
    try:
        response = requests.get(f"{tracking_uri}/health", timeout=5)
        if response.status_code != 200:
            print("MLflow not accessible, continuing without tracking")
            return None
    except:
        print("MLflow not accessible, continuing without tracking")
        return None
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        run_name = config['mlflow']['run_name']
        if run_name == 'auto':
            run_name = create_run_name()
        
        mlflow.start_run(run_name=run_name)
        
        mlflow.log_params({
            "framework": "SmartAudit",
            "model_type": "student",
            "model_name": config['model']['name'],
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['batch_size'],
            "num_epochs": config['training']['num_epochs'],
            "lora_r": config['lora']['r'],
            "lora_alpha": config['lora']['lora_alpha'],
            "vulnerable_weight": config['training']['vulnerable_weight'],
        })
        
        mlflow.set_tag("framework", "SmartAudit")
        mlflow.set_tag("model_category", "student")
        for key, value in config['mlflow']['tags'].items():
            mlflow.set_tag(key, value)
        
        return mlflow.active_run().info.run_id
        
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        mlflow.end_run()
        return None

def log_model_info(model, config: Dict):
    if mlflow.active_run():
        trainable_params, total_params = model.get_nb_trainable_parameters()
        
        mlflow.log_params({
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": round((trainable_params / total_params) * 100, 2),
            "model_size_category": "7B",
        })
        
        from .gpu_utils import get_gpu_info
        gpu_info = get_gpu_info()
        mlflow.log_params({
            "gpu_name": gpu_info['name'],
            "gpu_memory_gb": gpu_info['memory_gb'],
        })

def log_final_results(metrics_report: Dict, output_dir: str):
    if mlflow.active_run():
        final_metrics = metrics_report.get('final_metrics', {})
        
        key_metrics = {
            "final_f1_macro": final_metrics.get('eval_f1_macro', 0),
            "final_recall_vulnerable": final_metrics.get('eval_recall_vulnerable', 0),
            "final_accuracy": final_metrics.get('eval_accuracy', 0),
            "final_recall_machine_auditable": final_metrics.get('eval_recall_machine_auditable', 0),
            "final_recall_machine_unauditable": final_metrics.get('eval_recall_machine_unauditable', 0),
        }
        
        mlflow.log_metrics(key_metrics)
        
        # Log all metrics
        mlflow.log_metrics({
            f"final_{k}": v for k, v in final_metrics.items()
            if isinstance(v, (int, float)) and k not in key_metrics
        })
        
        # Log artifacts
        mlflow.log_dict(metrics_report, "metrics_report.json")
        mlflow.log_artifact("config.yaml")
        
        # Log confusion matrix plot if exists
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path)