from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from pathlib import Path
import sys

def load_datasets(config):
    # Check task type to determine data path
    task_type = config['metrics'].get('task_type', 'binary')
    
    if task_type == 'multiclass':
        data_path = Path('./processed_data')
    else:
        data_path = Path(config['data']['processed_data_path'])
    
    if not data_path.exists():
        print("ERROR: Processed data not found!")
        print(f"Expected path: {data_path}")
        if task_type == 'multiclass':
            print("Please run: python load_data_multiclass.py")
        else:
            print("Please run setup.sh first to prepare the data.")
        sys.exit(1)
    
    print(f"\nLoading {task_type} datasets from {data_path}...")
    train_dataset = load_from_disk(data_path / "train")
    eval_dataset = load_from_disk(data_path / "val")
    
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(eval_dataset):,}")
    
    return train_dataset, eval_dataset

def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )