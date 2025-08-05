# Utility functions for training
import torch
import os
import json
from datetime import datetime
import subprocess
import psutil
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, List, Tuple
import requests

def get_gpu_info():
    if not torch.cuda.is_available():
        return {
            "name": "No GPU",
            "memory_gb": 0,
            "recommended_batch_size": 1,
            "recommended_gradient_accumulation": 16
        }
    
    gpu_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # GPU presets
    if "A100" in gpu_name and memory_gb > 70:
        config = "a100-80gb"
        batch_size, grad_accum = 16, 1
    elif "A100" in gpu_name and memory_gb > 35:
        config = "a100-40gb"
        batch_size, grad_accum = 8, 2
    elif "A6000" in gpu_name and memory_gb > 45:
        config = "rtx-a6000-48gb"
        batch_size, grad_accum = 6, 3
    elif ("4090" in gpu_name or "3090" in gpu_name) and memory_gb > 20:
        config = "rtx-24gb"
        batch_size, grad_accum = 4, 4
    elif memory_gb > 14:
        config = "rtx-16gb"
        batch_size, grad_accum = 2, 8
    else:
        config = "default"
        batch_size, grad_accum = 1, 16
    
    return {
        "name": gpu_name,
        "memory_gb": round(memory_gb, 1),
        "config_name": config,
        "recommended_batch_size": batch_size,
        "recommended_gradient_accumulation": grad_accum
    }

def get_system_info():
    try:
        # cpu info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # mem info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1e9
        memory_percent = memory.percent
        
        # disk info
        disk = psutil.disk_usage('/')
        disk_gb = disk.total / 1e9
        disk_percent = disk.percent
        
        return {
            "cpu_cores": cpu_count,
            "cpu_usage": f"{cpu_percent}%",
            "memory_gb": round(memory_gb, 1),
            "memory_usage": f"{memory_percent}%",
            "disk_gb": round(disk_gb, 1),
            "disk_usage": f"{disk_percent}%"
        }
    except:
        return {}

def check_environment():
    print("Environment Check")
    print("=" * 50)
    
    import sys
    print(f"Python: {sys.version.split()[0]}")
    
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        gpu_info = get_gpu_info()
        print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")
        print(f"Recommended config: {gpu_info['config_name']}")
    
    # Sys info
    sys_info = get_system_info()
    if sys_info:
        print(f"\nSystem:")
        print(f"  CPU: {sys_info['cpu_cores']} cores ({sys_info['cpu_usage']})")
        print(f"  RAM: {sys_info['memory_gb']} GB ({sys_info['memory_usage']})")
        print(f"  Disk: {sys_info['disk_gb']} GB ({sys_info['disk_usage']})")
    
    print("=" * 50)

def find_latest_checkpoint(output_dir):
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

def create_run_name():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_info = get_gpu_info()
    gpu_name = gpu_info['name'].replace(" ", "-").lower()
    return f"ftaudit_{gpu_name}_{timestamp}"

def save_training_info(config, output_dir):
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
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

def test_mlflow_connection(tracking_uri):
    """Check mlflow server is accessible"""
    try:
        response = requests.get(f"{tracking_uri}/health", timeout=5)
        if response.status_code == 200:
            print(f"MLflow server is accessible at {tracking_uri}")
            return True
        else:
            print(f"MLflow server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to MLflow server at {tracking_uri}")
        print("   Make sure the server is running and accessible")
        return False
    except Exception as e:
        print(f"Error connecting to MLflow: {e}")
        return False

def get_mlflow_tracking_uri():
    import os
    # first check environment variable
    if os.environ.get('MLFLOW_TRACKING_URI'):
        return os.environ.get('MLFLOW_TRACKING_URI')
    
    # then check config file
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000')
    except:
        return 'http://localhost:5000'


class VulnerabilityMetrics:
    """Mtrics for vulnerability detection"""
    
    def __init__(self, tokenizer, vulnerable_token="vulnerable", safe_token="safe"):
        self.tokenizer = tokenizer
        self.vulnerable_token_id = tokenizer.encode(vulnerable_token, add_special_tokens=False)[0]
        self.safe_token_id = tokenizer.encode(safe_token, add_special_tokens=False)[0]
        
    def extract_predictions(self, predictions, labels):
        """extract vulnerability predictions from model outputs"""
        pred_labels = []
        true_labels = []
        
        # find positions where we need to predict vulnerability status
        for i, (pred_seq, label_seq) in enumerate(zip(predictions, labels)):
            # look for the last meaningful token position
            for j in range(len(label_seq)):
                if label_seq[j] in [self.vulnerable_token_id, self.safe_token_id]:
                    # get predicted token
                    pred_token = torch.argmax(pred_seq[j])
                    
                    # convert to binary labels
                    true_label = 1 if label_seq[j] == self.vulnerable_token_id else 0
                    pred_label = 1 if pred_token == self.vulnerable_token_id else 0
                    
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
                    break
        
        return np.array(pred_labels), np.array(true_labels)
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Classification metrics for vulnerability detection"""
        predictions, labels = eval_preds
        
        # handle different prediction formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # extract vulnerability predictions
        pred_labels, true_labels = self.extract_predictions(predictions, labels)
        
        if len(pred_labels) == 0:
            return {
                "eval_accuracy": 0.0,
                "eval_precision": 0.0,
                "eval_recall": 0.0,
                "eval_f1": 0.0,
            }
        
        # calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        
        # calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0
        )
        
        metrics = {
            "eval_accuracy": float(accuracy),
            "eval_precision": float(precision),
            "eval_recall": float(recall),
            "eval_f1": float(f1),
            # per-class metrics
            "eval_precision_safe": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
            "eval_precision_vulnerable": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
            "eval_recall_safe": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
            "eval_recall_vulnerable": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
            # counts
            "eval_total_samples": len(true_labels),
            "eval_vulnerable_samples": int(np.sum(true_labels)),
            "eval_safe_samples": int(np.sum(1 - true_labels)),
        }
        
        return metrics
    
    def compute_confusion_matrix(self, eval_preds) -> np.ndarray:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
            
        pred_labels, true_labels = self.extract_predictions(predictions, labels)
        
        if len(pred_labels) == 0:
            return np.array([[0, 0], [0, 0]])
            
        return confusion_matrix(true_labels, pred_labels)
    
    def generate_classification_report(self, eval_preds) -> str:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
            
        pred_labels, true_labels = self.extract_predictions(predictions, labels)
        
        if len(pred_labels) == 0:
            return "No predictions found"
            
        return classification_report(
            true_labels, pred_labels,
            target_names=['Safe', 'Vulnerable'],
            digits=4
        )


class MetricsCallback:
    """Callback to log detailed metrics during training"""
    
    def __init__(self, metrics_calculator, log_confusion_matrix=True):
        self.metrics_calculator = metrics_calculator
        self.log_confusion_matrix = log_confusion_matrix
        self.best_f1 = 0.0
        self.best_recall = 0.0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """called after evaluation"""
        if metrics:
            # track best metrics
            current_f1 = metrics.get('eval_f1', 0.0)
            current_recall = metrics.get('eval_recall_vulnerable', 0.0)
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                metrics['eval_best_f1'] = self.best_f1
                
            if current_recall > self.best_recall:
                self.best_recall = current_recall
                metrics['eval_best_recall_vulnerable'] = self.best_recall
            
            # log summary
            print(f"\nVulnerability Detection Metrics (Step {state.global_step}):")
            print(f"   Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            print(f"   Precision: {metrics.get('eval_precision', 0):.4f}")
            print(f"   Recall: {metrics.get('eval_recall', 0):.4f}")
            print(f"   F1 Score: {metrics.get('eval_f1', 0):.4f}")
            print(f"   Recall (Vulnerable): {metrics.get('eval_recall_vulnerable', 0):.4f}")
            print(f"   Total Samples: {metrics.get('eval_total_samples', 0)}")
            print(f"   Vulnerable: {metrics.get('eval_vulnerable_samples', 0)}")
            print(f"   Safe: {metrics.get('eval_safe_samples', 0)}")


def create_weighted_loss_function(vulnerable_weight=2.0):
    """Weighted loss function that penalizes missing vulnerabilities more"""
    def weighted_loss(logits, labels, vocab_size, vulnerable_token_id, safe_token_id):
        # standard cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # calculate per-token loss
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        # apply higher weight to vulnerable tokens
        weights = torch.ones_like(shift_labels, dtype=torch.float)
        weights[shift_labels == vulnerable_token_id] = vulnerable_weight
        
        # apply weights
        weighted_loss = loss.view(shift_labels.size()) * weights
        
        return weighted_loss.mean()
    
    return weighted_loss