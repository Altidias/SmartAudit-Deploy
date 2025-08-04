#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import gc
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
import json

from utils import (
    get_gpu_info,
    check_environment,
    find_latest_checkpoint,
    create_run_name,
    save_training_info
)

# Load environment variables
load_dotenv()

def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    gpu_info = get_gpu_info()
    if config['training']['batch_size'] == 'auto':
        config['training']['batch_size'] = gpu_info['recommended_batch_size']
    if config['training']['gradient_accumulation_steps'] == 'auto':
        config['training']['gradient_accumulation_steps'] = gpu_info['recommended_gradient_accumulation']
    
    if config['wandb']['run_name'] == 'auto':
        config['wandb']['run_name'] = create_run_name()
    
    return config

def setup_training_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    check_environment()

def main():
    setup_training_environment()
    
    print("\nLoading configuration...")
    config = load_config()
    
    save_training_info(config, config['paths']['output_dir'])
    
    use_wandb = config['training']['report_to'] == 'wandb' and os.environ.get('WANDB_API_KEY')
    if use_wandb:
        print("\n📊 Initializing W&B...")
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['run_name'],
            tags=config['wandb']['tags'],
            config=config
        )
    else:
        print("\nW&B disabled (no API key or disabled in config)")
        config['training']['report_to'] = 'none'
    
    # Load model
    print(f"\nLoading model: {config['model']['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=config['model']['trust_remote_code'],
        use_cache=config['model']['use_cache'],
        cache_dir=config['paths']['cache_dir']
    )
    
    # gradient checkpointing
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled")
    
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        cache_dir=config['paths']['cache_dir']
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nApplying LoRA...")
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        target_modules=config['lora']['target_modules'],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print("\nLoading datasets...")
    data_path = Path(config['data']['processed_data_path'])
    if not data_path.exists():
        print("Processed data not found! Please run setup.sh first.")
        sys.exit(1)
    
    train_dataset = load_from_disk(data_path / "train")
    eval_dataset = load_from_disk(data_path / "val")
    print(f"✅ Loaded {len(train_dataset)} training samples")
    print(f"✅ Loaded {len(eval_dataset)} validation samples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    print("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=config['paths']['output_dir'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        num_train_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        fp16=config['training']['fp16'],
        fp16_opt_level=config['training']['fp16_opt_level'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        max_grad_norm=config['training']['max_grad_norm'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=False,
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        report_to=config['training']['report_to'],
        run_name=config['wandb']['run_name'],
        push_to_hub=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=['labels']
    )
    
    checkpoint = find_latest_checkpoint(config['paths']['output_dir'])
    if checkpoint:
        print(f"\nResuming from checkpoint: {checkpoint}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # log final conf
    if use_wandb:
        gpu_info = get_gpu_info()
        wandb.config.update({
            "gpu_name": gpu_info['name'],
            "gpu_memory_gb": gpu_info['memory_gb'],
            "effective_batch_size": config['training']['batch_size'] * config['training']['gradient_accumulation_steps'],
            "total_train_samples": len(train_dataset),
            "total_eval_samples": len(eval_dataset),
            "max_length": config['data']['max_length']
        })
    
    print("\n" + "="*50)
    print("Starting training!")
    print(f"   Model: {config['model']['name']}")
    print(f"   GPU: {get_gpu_info()['name']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"   Total optimization steps: {len(train_dataset) // (config['training']['batch_size'] * config['training']['gradient_accumulation_steps']) * config['training']['num_epochs']}")
    print("="*50 + "\n")
    
    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        print("\nSaving final model...")
        trainer.save_model(os.path.join(config['paths']['output_dir'], "final_model"))
        tokenizer.save_pretrained(os.path.join(config['paths']['output_dir'], "final_model"))
        
        # save lora adapters separately
        model.save_pretrained(os.path.join(config['paths']['output_dir'], "lora_adapters"))
        
        # training results
        with open(os.path.join(config['paths']['output_dir'], "training_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        print("\nTraining completed successfully!")
        print(f"   Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
        print(f"   Models saved to: {config['paths']['output_dir']}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model(os.path.join(config['paths']['output_dir'], "interrupted_model"))
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        if use_wandb:
            wandb.finish()
        
        # cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()