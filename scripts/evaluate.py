#!/usr/bin/env python3

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.model import load_model_and_tokenizer
from src.data import load_datasets
from src.metrics import SmartAuditMetrics
from src.trainer import SmartAuditTrainer
from transformers import TrainingArguments
from peft import PeftModel

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SmartAudit student model")
    parser.add_argument("--model-path", type=str, 
                      default="output/student_model",
                      help="Path to model directory")
    parser.add_argument("--lora-path", type=str, default=None,
                      help="Path to LoRA adapters (if separate)")
    args = parser.parse_args()
    
    print("SmartAudit Model Evaluation")
    print("="*50)
    
    config = load_config()
    
    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(config)
    
    if args.lora_path:
        print(f"Loading LoRA adapters from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    _, eval_dataset = load_datasets(config)
    
    metrics_calc = SmartAuditMetrics(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=config['training']['batch_size'],
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=['labels']
    )
    
    trainer = SmartAuditTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics_calc.compute_metrics,
        vulnerable_weight=config['training']['vulnerable_weight']
    )
    
    print("\nRunning mevaluation...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  F1 Macro: {eval_results.get('eval_f1_macro', 0):.4f}")
    print(f"  Recall (Vulnerable): {eval_results.get('eval_recall_vulnerable', 0):.4f}")
    
    print(f"\nBy Vulnerability Category:")
    print(f"  Machine-Auditable Recall: {eval_results.get('eval_recall_machine_auditable', 0):.4f}")
    print(f"  Machine-Unauditable Recall: {eval_results.get('eval_recall_machine_unauditable', 0):.4f}")
    
    print(f"\nTop Vulnerability Types Performance:")
    for key, value in sorted(eval_results.items()):
        if key.startswith('eval_f1_') and not key.endswith('_macro'):
            vuln_type = key.replace('eval_f1_', '')
            if vuln_type != 'safe':
                print(f"  {vuln_type}: F1={value:.4f}")
    
    from src.config import check_readiness
    check_readiness(eval_results, config)
    
    results = {
        "framework": "SmartAudit",
        "model": config['model']['name'],
        "metrics": eval_results,
        "num_vulnerability_types": metrics_calc.num_classes,
        "production_ready": (
            eval_results.get('eval_recall_vulnerable', 0) >= config['metrics']['minimum_recall_vulnerable'] and
            eval_results.get('eval_f1_macro', 0) >= config['metrics']['minimum_f1_macro']
        )
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: evaluation_results.json")

if __name__ == "__main__":
    main()