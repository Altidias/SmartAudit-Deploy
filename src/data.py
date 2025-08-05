from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from pathlib import Path
import sys

def load_datasets(config):
    data_path = Path('./processed_data')
    
    if not data_path.exists():
        print("ERROR: processed data not found!")
        print(f"Expected path: {data_path}")
        print("Please ensure you have the dataset prepared.")
        sys.exit(1)
    
    print(f"\nLoading datasets from {data_path}...")
    train_dataset = load_from_disk(data_path / "train")
    eval_dataset = load_from_disk(data_path / "val")
    
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(eval_dataset):,}")
    
    # Verify vulnerability types file exists
    vuln_types_path = data_path / "vuln_types.json"
    if vuln_types_path.exists():
        import json
        with open(vuln_types_path, 'r') as f:
            vuln_types = json.load(f)
        print(f"  Vulnerability types: {len(vuln_types)}")
        print(f"  Machine-auditable types: ~10")
        print(f"  Machine-unauditable types: ~{len(vuln_types) - 11}")
    
    return train_dataset, eval_dataset

def get_data_collator(tokenizer):
    """Get data collator for training"""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )