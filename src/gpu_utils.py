import torch
import gc
import psutil
from typing import Dict, Tuple

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_gpu_info() -> Dict:
    if not torch.cuda.is_available():
        return {
            "name": "No GPU",
            "memory_gb": 0,
            "available_memory_gb": 0,
            "recommended_batch_size": 1,
            "recommended_gradient_accumulation": 16,
            "config_name": "cpu"
        }
    
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    available_memory = torch.cuda.mem_get_info(device)[0] / 1e9
    
    # Conservative batch size recommendations based on GPU type
    # For 7B models with 4096 sequence length
    gpu_configs = {
        "A100": {"batch_size": 4, "grad_accum": 4},
        "A6000": {"batch_size": 3, "grad_accum": 6},
        "V100": {"batch_size": 2, "grad_accum": 8},
        "4090": {"batch_size": 2, "grad_accum": 8},
        "3090": {"batch_size": 1, "grad_accum": 16},
        "A40": {"batch_size": 2, "grad_accum": 8},  # Reduced from 4
        "L40": {"batch_size": 2, "grad_accum": 8},
    }
    
    # Find matching config
    config = {"batch_size": 2, "grad_accum": 8}  # default
    for gpu_key, gpu_config in gpu_configs.items():
        if gpu_key in gpu_name:
            config = gpu_config
            break
    
    # Adjust based on available memory
    if available_memory < 20:
        config["batch_size"] = max(1, config["batch_size"] // 2)
        config["grad_accum"] = min(16, config["grad_accum"] * 2)
    
    return {
        "name": gpu_name,
        "memory_gb": round(total_memory, 1),
        "available_memory_gb": round(available_memory, 1),
        "config_name": f"{gpu_name.replace(' ', '-').lower()}-{int(total_memory)}gb",
        "recommended_batch_size": config["batch_size"],
        "recommended_gradient_accumulation": config["grad_accum"],
    }

def test_batch_size(model, tokenizer, batch_size: int, seq_length: int) -> bool:
    """Test if a batch size works without OOM"""
    try:
        clear_gpu_memory()
        
        # Create dummy input
        input_ids = torch.randint(
            0, tokenizer.vocab_size, 
            (batch_size, seq_length), 
            device='cuda',
            dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        
        # Create labels for loss computation
        labels = input_ids.clone()
        
        # Test forward pass with gradient computation and loss
        model.train()
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Also test backward pass
            loss.backward()
        
        # Clear gradients
        model.zero_grad()
        
        # Check memory usage
        used_memory = torch.cuda.memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Leave 20% headroom for safety (increased from 15%)
        if used_memory > total_memory * 0.80:
            return False
            
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False
        raise e
    finally:
        clear_gpu_memory()

def find_optimal_batch_size(model, tokenizer, max_length: int = 4096) -> Tuple[int, int]:
    """Find optimal batch size using the actual sequence length"""
    print(f"\nFinding optimal batch size for sequence length {max_length}...")
    
    # Start with conservative estimates based on model size
    model_params = sum(p.numel() for p in model.parameters()) / 1e9
    
    # More conservative for 7B models with long sequences
    if max_length >= 4096:
        if model_params > 60:  # 70B models
            test_sizes = [1]
        elif model_params > 6:   # 7B models
            test_sizes = [1, 2, 3, 4]
        else:  # Smaller models
            test_sizes = [1, 2, 4, 6, 8]
    else:
        if model_params > 60:  # 70B models
            test_sizes = [1]
        elif model_params > 10:  # 13B models
            test_sizes = [1, 2]
        elif model_params > 6:   # 7B models
            test_sizes = [1, 2, 4, 6, 8]
        else:  # Smaller models
            test_sizes = [2, 4, 8, 12, 16]
    
    optimal_batch_size = 1
    
    # Test with actual sequence length
    for batch_size in test_sizes:
        print(f"Testing batch size {batch_size}...", end=" ")
        if test_batch_size(model, tokenizer, batch_size, max_length):
            optimal_batch_size = batch_size
            print("✓")
        else:
            print("✗ (OOM)")
            break
    
    # Calculate gradient accumulation
    target_effective_batch_size = 16
    grad_accum = max(1, target_effective_batch_size // optimal_batch_size)
    
    print(f"\nOptimal configuration:")
    print(f"  Batch size: {optimal_batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {optimal_batch_size * grad_accum}")
    
    return optimal_batch_size, grad_accum

def get_system_info() -> Dict:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_cores": psutil.cpu_count(),
        "cpu_usage": f"{psutil.cpu_percent(interval=1)}%",
        "memory_gb": round(memory.total / 1e9, 1),
        "memory_usage": f"{memory.percent}%",
        "disk_gb": round(disk.total / 1e9, 1),
        "disk_usage": f"{disk.percent}%"
    }