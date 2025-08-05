import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, Tuple
from transformers import TrainerCallback

class VulnerabilityMetrics:
    def __init__(self, tokenizer, vulnerable_token="vulnerable", safe_token="safe"):
        self.tokenizer = tokenizer
        self.vulnerable_token_id = tokenizer.encode(vulnerable_token, add_special_tokens=False)[0]
        self.safe_token_id = tokenizer.encode(safe_token, add_special_tokens=False)[0]
    
    def extract_predictions(self, predictions, labels) -> Tuple[np.ndarray, np.ndarray]:
        pred_labels = []
        true_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            for j in range(len(label_seq)):
                if label_seq[j] in [self.vulnerable_token_id, self.safe_token_id]:
                    pred_token = torch.argmax(pred_seq[j])
                    true_label = 1 if label_seq[j] == self.vulnerable_token_id else 0
                    pred_label = 1 if pred_token == self.vulnerable_token_id else 0
                    
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
                    break
        
        return np.array(pred_labels), np.array(true_labels)
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        predictions, labels = eval_preds
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        pred_labels, true_labels = self.extract_predictions(predictions, labels)
        
        if len(pred_labels) == 0:
            return {
                "eval_accuracy": 0.0,
                "eval_precision": 0.0,
                "eval_recall": 0.0,
                "eval_f1": 0.0,
            }
        
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0
        )
        
        return {
            "eval_accuracy": float(accuracy),
            "eval_precision": float(precision),
            "eval_recall": float(recall),
            "eval_f1": float(f1),
            "eval_precision_safe": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
            "eval_precision_vulnerable": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
            "eval_recall_safe": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
            "eval_recall_vulnerable": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
            "eval_total_samples": len(true_labels),
            "eval_vulnerable_samples": int(np.sum(true_labels)),
            "eval_safe_samples": int(np.sum(1 - true_labels)),
        }
    
    def compute_confusion_matrix(self, eval_preds) -> np.ndarray:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        pred_labels, true_labels = self.extract_predictions(predictions, labels)
        
        if len(pred_labels) == 0:
            return np.array([[0, 0], [0, 0]])
        
        return confusion_matrix(true_labels, pred_labels)
    
    def generate_classification_report(self, eval_preds) -> str:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        pred_labels, true_labels = self.extract_predictions(predictions, labels)
        
        if len(pred_labels) == 0:
            return "No predictions found"
        
        return classification_report(
            true_labels, pred_labels,
            target_names=['Safe', 'Vulnerable'],
            digits=4
        )

class MetricsCallback(TrainerCallback):
    def __init__(self, metrics_calculator, log_confusion_matrix=True):
        super().__init__()
        self.metrics_calculator = metrics_calculator
        self.log_confusion_matrix = log_confusion_matrix
        self.best_f1 = 0.0
        self.best_recall = 0.0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            current_f1 = metrics.get('eval_f1', 0.0)
            current_recall = metrics.get('eval_recall_vulnerable', 0.0)
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                metrics['eval_best_f1'] = self.best_f1
            
            if current_recall > self.best_recall:
                self.best_recall = current_recall
                metrics['eval_best_recall_vulnerable'] = self.best_recall
            
            print(f"\nMetrics at step {state.global_step}:")
            print(f"  F1: {metrics.get('eval_f1', 0):.4f}")
            print(f"  Recall (Vulnerable): {metrics.get('eval_recall_vulnerable', 0):.4f}")
            print(f"  Precision: {metrics.get('eval_precision', 0):.4f}")

def generate_metrics_report(trainer, eval_dataset, metrics_calc, output_dir):
    import json
    import os
    
    eval_results = trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    
    confusion_matrix = metrics_calc.compute_confusion_matrix(predictions)
    classification_report = metrics_calc.generate_classification_report(predictions)
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Safe  Vulnerable")
    print(f"Actual Safe      {confusion_matrix[0, 0]:<5} {confusion_matrix[0, 1]:<5}")
    print(f"Actual Vulnerable {confusion_matrix[1, 0]:<5} {confusion_matrix[1, 1]:<5}")
    
    print("\n" + classification_report)
    
    report = {
        "final_metrics": eval_results,
        "confusion_matrix": confusion_matrix.tolist(),
        "classification_report": classification_report,
    }
    
    with open(os.path.join(output_dir, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def save_confusion_matrix_plot(confusion_matrix, output_dir):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Safe', 'Vulnerable'],
                    yticklabels=['Safe', 'Vulnerable'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        return cm_path
    except:
        return None