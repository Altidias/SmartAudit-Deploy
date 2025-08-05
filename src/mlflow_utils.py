import mlflow
import mlflow.pytorch
import requests
import os
from datetime import datetime
from transformers import TrainerCallback
from typing import Dict, Optional

class MLflowCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_metrics = {k: v for k, v in metrics.items() 
                          if isinstance(v, (int, float)) and k.startswith('eval_')}
            if eval_metrics:
                mlflow.log_metrics(eval_metrics, step=state.global_step)

def test_mlflow_connection(tracking_uri: str) -> bool:
    try:
        response = requests.get(f"{tracking_uri}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ MLflow server accessible at {tracking_uri}")
            return True
        else:
            print(f"✗ MLflow server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to MLflow at {tracking_uri}")
        return False
    except Exception as e:
        print(f"✗ MLflow connection error: {e}")
        return False

def setup_mlflow(config: Dict) -> Optional[str]:
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
    
    print(f"\nSetting up MLflow tracking...")
    
    if not test_mlflow_connection(tracking_uri):
        print("Continuing without MLflow tracking")
        return None
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        run_name = config['mlflow']['run_name']
        if run_name == 'auto':
            run_name = create_run_name()
        
        mlflow.start_run(run_name=run_name)
        
        # Log basic parameters
        mlflow.log_params({
            "model_name": config['model']['name'],
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['batch_size'],
            "num_epochs": config['training']['num_epochs'],
            "lora_r": config['lora']['r'],
            "vulnerable_weight": config['training']['vulnerable_weight'],
        })
        
        # Set tags
        for key, value in config['mlflow']['tags'].items():
            mlflow.set_tag(key, value)
        
        return mlflow.active_run().info.run_id
        
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        mlflow.end_run()
        return None

def create_run_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Avoid circular import
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "-").lower()[:20]
    else:
        gpu_name = "cpu"
    
    return f"vuln-detect_{gpu_name}_{timestamp}"

def log_model_info(model, config: Dict):
    if mlflow.active_run():
        trainable_params, total_params = model.get_nb_trainable_parameters()
        mlflow.log_params({
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": round((trainable_params / total_params) * 100, 2)
        })
        
        from .gpu_utils import get_gpu_info
        gpu_info = get_gpu_info()
        mlflow.log_params({
            "gpu_name": gpu_info['name'],
            "gpu_memory_gb": gpu_info['memory_gb'],
        })

def log_final_results(metrics_report: Dict, output_dir: str):
    if mlflow.active_run():
        # Log final metrics
        final_metrics = metrics_report.get('final_metrics', {})
        mlflow.log_metrics({
            f"final_{k}": v for k, v in final_metrics.items()
            if isinstance(v, (int, float))
        })
        
        # Log artifacts
        mlflow.log_dict(metrics_report, "metrics_report.json")
        mlflow.log_artifact("config.yaml")
        
        # Log confusion matrix plot if it exists
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path)

def end_mlflow_run(status: str = "FINISHED"):
    if mlflow.active_run():
        mlflow.log_param("training_status", status)
        mlflow.end_run()