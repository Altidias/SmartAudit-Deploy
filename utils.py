# utils.py - Utility functions for training
import torch
import os
import json
from datetime import datetime
import subprocess
import psutil
import sys

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
        batch_size, grad_accum = 10, 2
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
    """Get system information"""
    try:
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1e9
        memory_percent = memory.percent
        
        # Disk info
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
    
    # Python version
    import sys
    print(f"Python: {sys.version.split()[0]}")
    
    # PyTorch version
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        gpu_info = get_gpu_info()
        print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")
        print(f"Recommended config: {gpu_info['config_name']}")
    
    # System info
    sys_info = get_system_info()
    if sys_info:
        print(f"\nSystem:")
        print(f"  CPU: {sys_info['cpu_cores']} cores ({sys_info['cpu_usage']})")
        print(f"  RAM: {sys_info['memory_gb']} GB ({sys_info['memory_usage']})")
        print(f"  Disk: {sys_info['disk_gb']} GB ({sys_info['disk_usage']})")
    
    print("=" * 50)

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output directory"""
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