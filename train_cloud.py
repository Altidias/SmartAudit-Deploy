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
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import mlflow
import mlflow.pytorch
from datasets import load_from_disk
from dotenv import load_dotenv
import json

from utils import (
    get_gpu_info,
    check_environment,
    find_latest_checkpoint,
    create_run_name,
    save_training_info,
    test_mlflow_connection,
    VulnerabilityMetrics,
    MetricsCallback,
    create_weighted_loss_function
)

# load environment variables
load_dotenv()


class MLflowCallback(TrainerCallback):
    """custom callback to log metrics to mlflow during training"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # log metrics to mlflow
            metrics = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
            
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # log evaluation metrics
            eval_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key.startswith('eval_'):
                    eval_metrics[key] = value
            
            if eval_metrics:
                mlflow.log_metrics(eval_metrics, step=state.global_step)


class VulnerabilityDetectionTrainer(Trainer):
    def __init__(self, *args, vulnerable_weight=2.0, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vulnerable_weight = vulnerable_weight
        self.vulnerable_token_id = tokenizer.encode("vulnerable", add_special_tokens=False)[0]
        self.safe_token_id = tokenizer.encode("safe", add_special_tokens=False)[0]
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        weighted loss for vulnerability detection
        
        args:
            model: the model
            inputs: iinput dictionary
            return_outputs: whether to return outputs along with loss
            num_items_in_batch: number of items in the batch
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # use weighted loss
        loss = create_weighted_loss_function(self.vulnerable_weight)(
            logits, labels, 
            self.model.config.vocab_size,
            self.vulnerable_token_id,
            self.safe_token_id
        )
        
        return (loss, outputs) if return_outputs else loss


def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    gpu_info = get_gpu_info()
    if config['training']['batch_size'] == 'auto':
        config['training']['batch_size'] = gpu_info['recommended_batch_size']
    if config['training']['gradient_accumulation_steps'] == 'auto':
        config['training']['gradient_accumulation_steps'] = gpu_info['recommended_gradient_accumulation']
    
    if config['mlflow']['run_name'] == 'auto':
        config['mlflow']['run_name'] = create_run_name()
    
    return config


def setup_training_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # clear gpu cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    check_environment()


def main():
    setup_training_environment()
    
    print("\nLoading configuration...")
    config = load_config()
    
    save_training_info(config, config['paths']['output_dir'])
    
    # setup mlflow
    print(f"\nSetting up MLflow...")
    
    # override tracking uri from environment if set
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
    config['mlflow']['tracking_uri'] = tracking_uri
    
    print(f"   Tracking URI: {tracking_uri}")
    
    mlflow_available = False
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        # start mlflow run
        mlflow.start_run(run_name=config['mlflow']['run_name'])
        
        # log parameters
        mlflow.log_params({
            "model_name": config['model']['name'],
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['batch_size'],
            "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
            "num_epochs": config['training']['num_epochs'],
            "max_length": config['data']['max_length'],
            "lora_r": config['lora']['r'],
            "lora_alpha": config['lora']['lora_alpha'],
            "lora_dropout": config['lora']['lora_dropout'],
            "vulnerable_weight": config['training']['vulnerable_weight'],
        })
        
        # log gpu info
        gpu_info = get_gpu_info()
        mlflow.log_params({
            "gpu_name": gpu_info['name'],
            "gpu_memory_gb": gpu_info['memory_gb'],
            "gpu_config": gpu_info['config_name'],
        })
        
        # set tags
        for key, value in config['mlflow']['tags'].items():
            mlflow.set_tag(key, value)
        
        print("MLflow tracking initialized")
        mlflow_available = True
        
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        print("Continuing without MLflow tracking...")
        mlflow.end_run()
    
    # load model
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
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        cache_dir=config['paths']['cache_dir']
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # initialize metrics calculator
    print("Initializing metrics calculator...")
    metrics_calc = VulnerabilityMetrics(
        tokenizer,
        vulnerable_token=config['metrics']['vulnerable_token'],
        safe_token=config['metrics']['safe_token']
    )
    
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

    if config['training']['batch_size'] == 'auto':
        print("\nFinding optimal batch size...")
        try:
            from utils import find_optimal_batch_size
            optimal_batch_size, optimal_grad_accum = find_optimal_batch_size(
                model, tokenizer, 
                max_length=config['data']['max_length'],
                starting_batch_size=8 
            )
            config['training']['batch_size'] = optimal_batch_size
            config['training']['gradient_accumulation_steps'] = optimal_grad_accum
        except:
            # fb
            gpu_info = get_gpu_info()
            config['training']['batch_size'] = gpu_info['recommended_batch_size']
            config['training']['gradient_accumulation_steps'] = gpu_info['recommended_gradient_accumulation']
    
    # log model info to mlflow
    if mlflow_available and mlflow.active_run():
        trainable_params, total_params = model.get_nb_trainable_parameters()
        mlflow.log_params({
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": (trainable_params / total_params) * 100
        })
    
    # load datasets
    print("\nLoading datasets...")
    data_path = Path(config['data']['processed_data_path'])
    if not data_path.exists():
        print("Processed data not found! Please run setup.sh first.")
        sys.exit(1)
    
    train_dataset = load_from_disk(data_path / "train")
    eval_dataset = load_from_disk(data_path / "val")
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(eval_dataset)} validation samples")
    
    # log dataset info
    if mlflow_available and mlflow.active_run():
        mlflow.log_params({
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
        })
    
    # data collator
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
        greater_is_better=config['training']['greater_is_better'],
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        report_to="none",
        push_to_hub=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=['labels']
    )
    
    checkpoint = find_latest_checkpoint(config['paths']['output_dir'])
    if checkpoint:
        print(f"\nResuming from checkpoint: {checkpoint}")
    
    # create trainer with callbacks
    callbacks = []
    if mlflow_available and mlflow.active_run():
        callbacks.append(MLflowCallback())
    
    # add metrics callback
    metrics_callback = MetricsCallback(
        metrics_calc,
        log_confusion_matrix=config['metrics']['log_confusion_matrix']
    )
    callbacks.append(metrics_callback)
    
    # use custom trainer for weighted loss
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
    
    # log additional info
    if mlflow_available and mlflow.active_run():
        effective_batch_size = config['training']['batch_size'] * config['training']['gradient_accumulation_steps']
        total_steps = len(train_dataset) // effective_batch_size * config['training']['num_epochs']
        
        mlflow.log_params({
            "effective_batch_size": effective_batch_size,
            "total_optimization_steps": total_steps,
        })
        
        # log config file as artifact
        mlflow.log_artifact("config.yaml")
    
    print("\n" + "="*50)
    print("Starting training!")
    print(f"   Model: {config['model']['name']}")
    print(f"   GPU: {get_gpu_info()['name']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"   Total optimization steps: {len(train_dataset) // (config['training']['batch_size'] * config['training']['gradient_accumulation_steps']) * config['training']['num_epochs']}")
    if mlflow_available and mlflow.active_run():
        print(f"   MLflow Run ID: {mlflow.active_run().info.run_id}")
    print("="*50 + "\n")
    
    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        print("\nSaving final model...")
        trainer.save_model(os.path.join(config['paths']['output_dir'], "final_model"))
        tokenizer.save_pretrained(os.path.join(config['paths']['output_dir'], "final_model"))
        
        # save lora adapters separately
        model.save_pretrained(os.path.join(config['paths']['output_dir'], "lora_adapters"))
        
        # generate classification report
        print("\nGenerating classification report...")
        eval_results = trainer.evaluate()
        
        # get predictions for confusion matrix
        predictions = trainer.predict(eval_dataset)
        confusion_matrix = metrics_calc.compute_confusion_matrix(predictions)
        classification_report = metrics_calc.generate_classification_report(predictions)
        
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Safe  Vulnerable")
        print(f"Actual Safe      {confusion_matrix[0, 0]:<5} {confusion_matrix[0, 1]:<5}")
        print(f"Actual Vulnerable {confusion_matrix[1, 0]:<5} {confusion_matrix[1, 1]:<5}")
        
        print("\nDetailed Classification Report:")
        print(classification_report)
        
        # save metrics report
        metrics_report = {
            "final_metrics": eval_results,
            "confusion_matrix": confusion_matrix.tolist(),
            "classification_report": classification_report,
            "best_f1": metrics_callback.best_f1,
            "best_recall_vulnerable": metrics_callback.best_recall,
            "training_results": train_result.metrics
        }
        
        with open(os.path.join(config['paths']['output_dir'], "metrics_report.json"), "w") as f:
            json.dump(metrics_report, f, indent=2)
        
        # log to mlflow
        if mlflow_available and mlflow.active_run():
            # log final metrics
            final_metrics = {
                "final_train_loss": train_result.metrics.get('train_loss', 0),
                "final_eval_loss": eval_results.get('eval_loss', 0),
                "final_eval_f1": eval_results.get('eval_f1', 0),
                "final_eval_recall_vulnerable": eval_results.get('eval_recall_vulnerable', 0),
                "total_train_runtime": train_result.metrics.get('train_runtime', 0),
                "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
            }
            mlflow.log_metrics(final_metrics)
            
            # log artifacts
            mlflow.log_dict(metrics_report, "metrics_report.json")
            mlflow.log_artifacts(os.path.join(config['paths']['output_dir'], "lora_adapters"), "lora_adapters")
            
            # log confusion matrix as image
            if config['metrics']['log_confusion_matrix']:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Safe', 'Vulnerable'],
                                yticklabels=['Safe', 'Vulnerable'])
                    plt.title('Confusion Matrix')
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    plt.tight_layout()
                    cm_path = os.path.join(config['paths']['output_dir'], 'confusion_matrix.png')
                    plt.savefig(cm_path)
                    mlflow.log_artifact(cm_path)
                    plt.close()
                except:
                    pass
            
            # log model
            print("Logging model to MLflow...")
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                requirements_file="requirements.txt",
                extra_files=["config.yaml"],
            )
        
        # check production readiness
        print("\n" + "="*50)
        print("PRODUCTION READINESS CHECK")
        print("="*50)
        
        recall_vuln = eval_results.get('eval_recall_vulnerable', 0)
        precision = eval_results.get('eval_precision', 0)
        
        if recall_vuln >= config['metrics']['minimum_recall'] and precision >= config['metrics']['minimum_precision']:
            print("Model is ready for production use!")
        elif recall_vuln >= config['metrics']['minimum_recall']:
            print(f"Good recall but low precision - suitable as pre-filter")
            print(f"   Current precision: {precision:.4f}, target: {config['metrics']['minimum_precision']}+")
        else:
            print("Model needs more training")
            print(f"   Current recall: {recall_vuln:.4f}, target: {config['metrics']['minimum_recall']}+")
        
        print("\nTraining completed successfully!")
        print(f"   Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
        print(f"   Models saved to: {config['paths']['output_dir']}")
        if mlflow_available and mlflow.active_run():
            run_info = mlflow.active_run().info
            print(f"   MLflow Run: {run_info.run_id}")
            print(f"   View at: {config['mlflow']['tracking_uri']}/#/experiments/{run_info.experiment_id}/runs/{run_info.run_id}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model(os.path.join(config['paths']['output_dir'], "interrupted_model"))
        if mlflow_available and mlflow.active_run():
            mlflow.log_param("training_status", "interrupted")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if mlflow_available and mlflow.active_run():
            mlflow.log_param("training_status", "failed")
            mlflow.log_param("error", str(e))
        raise
    finally:
        if mlflow_available and mlflow.active_run():
            mlflow.end_run()
        
        # cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()