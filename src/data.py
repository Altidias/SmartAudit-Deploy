from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from pathlib import Path
import sys

def load_datasets(config):
    data_path = Path(config['data']['processed_data_path'])
    
    if not data_path.exists():
        print("ERROR: Processed data not found!")
        print(f"Expected path: {data_path}")
        print("Please run setup.sh first to prepare the data.")
        sys.exit(1)
    
    print("\nLoading datasets...")
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