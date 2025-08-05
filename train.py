#!/usr/bin/env python3
import sys
import torch
from pathlib import Path
from transformers import TrainingArguments

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import load_config, save_training_info, find_latest_checkpoint, setup_environment, check_production_readiness
from src.model import load_model_and_tokenizer, setup_lora, save_model_and_adapters
from src.data import load_datasets, get_data_collator
from src.trainer import VulnerabilityDetectionTrainer
from src.metrics import VulnerabilityMetrics, MetricsCallback, generate_metrics_report, save_confusion_matrix_plot
from src.mlflow_utils import setup_mlflow, MLflowCallback, log_model_info, log_final_results, end_mlflow_run
from src.gpu_utils import clear_gpu_memory, find_optimal_batch_size, get_gpu_info

def main():
    setup_environment()
    clear_gpu_memory()
    
    print("="*50)
    print("Vulnerability Detection Training")
    print("="*50)
    
    # Load configuration
    config = load_config()
    save_training_info(config, config['paths']['output_dir'])
    
    # Setup MLflow
    mlflow_run_id = setup_mlflow(config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    model = setup_lora(model, config)
    
    # Find optimal batch size if needed
    if config['training']['batch_size'] == 'auto':
        batch_size, grad_accum = find_optimal_batch_size(
            model, tokenizer, config['data']['max_length']
        )
        config['training']['batch_size'] = batch_size
        config['training']['gradient_accumulation_steps'] = grad_accum
    
    # Log model info
    log_model_info(model, config)
    
    # Load datasets
    train_dataset, eval_dataset = load_datasets(config)
    data_collator = get_data_collator(tokenizer)
    
    # Setup metrics
    metrics_calc = VulnerabilityMetrics(
        tokenizer,
        vulnerable_token=config['metrics']['vulnerable_token'],
        safe_token=config['metrics']['safe_token']
    )
    
    # Setup training arguments
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
        greater_is_better=config['training']['greater_is_better'],
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        report_to="none",
        push_to_hub=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=['labels']
    )
    
    # Setup callbacks
    callbacks = []
    if mlflow_run_id:
        callbacks.append(MLflowCallback())
    callbacks.append(MetricsCallback(metrics_calc))
    
    # Check for checkpoint
    checkpoint = find_latest_checkpoint(config['paths']['output_dir'])
    if checkpoint:
        print(f"\n✓ Resuming from checkpoint: {checkpoint}")
    
    # Create trainer
    trainer = VulnerabilityDetectionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=metrics_calc.compute_metrics,
        vulnerable_weight=config['training']['vulnerable_weight']
    )
    
    # Training info
    gpu_info = get_gpu_info()
    effective_batch_size = config['training']['batch_size'] * config['training']['gradient_accumulation_steps']
    total_steps = len(train_dataset) // effective_batch_size * config['training']['num_epochs']
    
    print("\n" + "="*50)
    print("Training Configuration")
    print("="*50)
    print(f"Model: {config['model']['name']}")
    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Total steps: {total_steps}")
    if mlflow_run_id:
        print(f"MLflow Run ID: {mlflow_run_id}")
    print("="*50 + "\n")
    
    try:
        # Train
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save model
        print("\nSaving model...")
        save_model_and_adapters(model, tokenizer, config['paths']['output_dir'])
        
        # Generate metrics report
        metrics_report = generate_metrics_report(
            trainer, eval_dataset, metrics_calc, config['paths']['output_dir']
        )
        
        # Save confusion matrix plot
        if config['metrics']['log_confusion_matrix']:
            cm_path = save_confusion_matrix_plot(
                metrics_calc.compute_confusion_matrix(trainer.predict(eval_dataset)),
                config['paths']['output_dir']
            )
        
        # Log final results to MLflow
        log_final_results(metrics_report, config['paths']['output_dir'])
        
        # Check production readiness
        check_production_readiness(metrics_report['final_metrics'], config)
        
        print("\n✓ Training completed successfully!")
        print(f"  Models saved to: {config['paths']['output_dir']}")
        
        end_mlflow_run("FINISHED")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        save_model_and_adapters(model, tokenizer, config['paths']['output_dir'])
        end_mlflow_run("INTERRUPTED")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        end_mlflow_run("FAILED")
        raise
        
    finally:
        clear_gpu_memory()

if __name__ == "__main__":
    main()