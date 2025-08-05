#!/usr/bin/env python3
"""Evaluate a trained model"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.model import load_model_and_tokenizer
from src.data import load_datasets
from src.metrics import VulnerabilityMetrics
from src.trainer import VulnerabilityDetectionTrainer
from transformers import TrainingArguments
from peft import PeftModel

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate vulnerability detection model")
    parser.add_argument("--model-path", type=str, default="output/final_model",
                      help="Path to model directory")
    parser.add_argument("--lora-path", type=str, default=None,
                      help="Path to LoRA adapters (if separate)")
    args = parser.parse_args()
    
    print("Model Evaluation")
    print("="*50)
    
    # Load config
    config = load_config()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(config)
    
    if args.lora_path:
        print(f"Loading LoRA adapters from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    # Load evaluation dataset
    _, eval_dataset = load_datasets(config)
    
    # Setup metrics
    metrics_calc = VulnerabilityMetrics(
        tokenizer,
        vulnerable_token=config['metrics']['vulnerable_token'],
        safe_token=config['metrics']['safe_token']
    )
    
    # Create minimal trainer for evaluation
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=config['training']['batch_size'],
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=['labels']
    )
    
    trainer = VulnerabilityDetectionTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics_calc.compute_metrics,
        vulnerable_weight=config['training']['vulnerable_weight']
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    eval_results = trainer.evaluate()
    
    # Generate detailed report
    predictions = trainer.predict(eval_dataset)
    confusion_matrix = metrics_calc.compute_confusion_matrix(predictions)
    classification_report = metrics_calc.generate_classification_report(predictions)
    
    # Display results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  Precision: {eval_results.get('eval_precision', 0):.4f}")
    print(f"  Recall: {eval_results.get('eval_recall', 0):.4f}")
    print(f"  F1 Score: {eval_results.get('eval_f1', 0):.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Safe contracts:")
    print(f"    Precision: {eval_results.get('eval_precision_safe', 0):.4f}")
    print(f"    Recall: {eval_results.get('eval_recall_safe', 0):.4f}")
    print(f"  Vulnerable contracts:")
    print(f"    Precision: {eval_results.get('eval_precision_vulnerable', 0):.4f}")
    print(f"    Recall: {eval_results.get('eval_recall_vulnerable', 0):.4f}")
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Safe  Vulnerable")
    print(f"Actual Safe      {confusion_matrix[0, 0]:<5} {confusion_matrix[0, 1]:<5}")
    print(f"Actual Vulnerable {confusion_matrix[1, 0]:<5} {confusion_matrix[1, 1]:<5}")
    
    print("\nDetailed Classification Report:")
    print(classification_report)
    
    # Production readiness check
    recall_vuln = eval_results.get('eval_recall_vulnerable', 0)
    precision = eval_results.get('eval_precision', 0)
    
    print("\n" + "="*50)
    print("Production Readiness")
    print("="*50)
    
    if recall_vuln >= config['metrics']['minimum_recall'] and precision >= config['metrics']['minimum_precision']:
        print("✓ Model is production ready!")
    elif recall_vuln >= config['metrics']['minimum_recall']:
        print("⚠ Good recall but needs better precision for production")
    else:
        print("✗ Model needs more training to meet production criteria")
    
    print(f"\nTarget thresholds:")
    print(f"  Minimum recall (vulnerable): {config['metrics']['minimum_recall']}")
    print(f"  Minimum precision: {config['metrics']['minimum_precision']}")
    
    # Save results
    results = {
        "metrics": eval_results,
        "confusion_matrix": confusion_matrix.tolist(),
        "classification_report": classification_report,
        "production_ready": recall_vuln >= config['metrics']['minimum_recall'] and precision >= config['metrics']['minimum_precision']
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: evaluation_results.json")

if __name__ == "__main__":
    main()