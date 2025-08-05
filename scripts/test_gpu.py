#!/usr/bin/env python3
"""Test GPU memory and find optimal batch size"""

import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.gpu_utils import get_gpu_info, test_batch_size, clear_gpu_memory
from src.config import load_config
from src.model import load_model_and_tokenizer, setup_lora

def main():
    if not torch.cuda.is_available():
        print("No GPU available!")
        return
    
    print("GPU Memory Test")
    print("="*50)
    
    # Get GPU info
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"Total Memory: {gpu_info['memory_gb']} GB")
    print(f"Available Memory: {gpu_info['available_memory_gb']} GB")
    print()
    
    # Load config
    config = load_config()
    seq_length = config['data']['max_length']
    
    print(f"Testing with sequence length: {seq_length}")
    print()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(config)
    model = setup_lora(model, config)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 6, 8, 12, 16]
    results = []
    
    print("\nTesting batch sizes:")
    print("-"*40)
    
    for batch_size in batch_sizes:
        print(f"Batch size {batch_size}: ", end="", flush=True)
        
        success = test_batch_size(model, tokenizer, batch_size, seq_length)
        results.append((batch_size, success))
        
        if success:
            print("✓ Success")
        else:
            print("✗ OOM")
            break
    
    print("-"*40)
    
    # Find optimal
    optimal = max([bs for bs, success in results if success], default=1)
    print(f"\nRecommended batch size: {optimal}")
    
    # Calculate gradient accumulation
    target_effective = 16
    grad_accum = max(1, target_effective // optimal)
    print(f"Recommended gradient accumulation: {grad_accum}")
    print(f"Effective batch size: {optimal * grad_accum}")
    
    # Memory usage estimate
    if optimal > 1:
        clear_gpu_memory()
        test_batch_size(model, tokenizer, optimal, seq_length)
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nEstimated memory usage: {used:.1f}/{total:.1f} GB ({used/total*100:.0f}%)")

if __name__ == "__main__":
    main()