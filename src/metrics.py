import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, Tuple
from transformers import TrainerCallback

class VulnerabilityMetrics:
    def __init__(self, tokenizer, vulnerable_token="vulnerable", safe_token="safe", task_type="binary", vuln_types=None):
        self.tokenizer = tokenizer
        self.task_type = task_type
        
        if task_type == "binary":
            self.vulnerable_token_id = tokenizer.encode(vulnerable_token, add_special_tokens=False)[0]
            self.safe_token_id = tokenizer.encode(safe_token, add_special_tokens=False)[0]
        elif task_type == "multiclass":
            # Load vulnerability types
            if vuln_types is None:
                import json
                import os
                if os.path.exists('./processed_data_multiclass/vuln_types.json'):
                    with open('./processed_data_multiclass/vuln_types.json', 'r') as f:
                        vuln_types = json.load(f)
                else:
                    # Fallback to binary if types not found
                    self.task_type = "binary"
                    self.vulnerable_token_id = tokenizer.encode(vulnerable_token, add_special_tokens=False)[0]
                    self.safe_token_id = tokenizer.encode(safe_token, add_special_tokens=False)[0]
                    return
            
            self.vuln_types = vuln_types
            self.type_to_id = {v: i for i, v in enumerate(vuln_types)}
            self.vuln_token_ids = {}
            
            # Get token IDs for each vulnerability type
            for vuln_type in vuln_types:
                tokens = tokenizer.encode(vuln_type, add_special_tokens=False)
                if tokens:
                    self.vuln_token_ids[vuln_type] = tokens[0]
    
    def extract_predictions(self, predictions, labels) -> Tuple[np.ndarray, np.ndarray]:
        pred_labels = []
        true_labels = []
        
        if self.task_type == "binary":
            # Binary classification logic
            for pred_seq, label_seq in zip(predictions, labels):
                for j in range(len(label_seq)):
                    if label_seq[j] in [self.vulnerable_token_id, self.safe_token_id]:
                        pred_token = torch.argmax(pred_seq[j])
                        true_label = 1 if label_seq[j] == self.vulnerable_token_id else 0
                        pred_label = 1 if pred_token == self.vulnerable_token_id else 0
                        
                        true_labels.append(true_label)
                        pred_labels.append(pred_label)
                        break
        
        elif self.task_type == "multiclass":
            # Multi-class classification logic
            for pred_seq, label_seq in zip(predictions, labels):
                found = False
                for j in range(len(label_seq)):
                    # Check if this position has a vulnerability type token
                    for vuln_type, token_id in self.vuln_token_ids.items():
                        if label_seq[j] == token_id:
                            # Found a vulnerability type token
                            pred_token = torch.argmax(pred_seq[j]).item()
                            
                            # Find which class the predicted token belongs to
                            pred_class = 0  # Default to first class (usually 'safe')
                            for vtype, tid in self.vuln_token_ids.items():
                                if pred_token == tid:
                                    pred_class = self.type_to_id[vtype]
                                    break
                            
                            true_class = self.type_to_id[vuln_type]
                            
                            pred_labels.append(pred_class)
                            true_labels.append(true_class)
                            found = True
                            break
                    
                    if found:
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
        
        if self.task_type == "binary":
            # Binary metrics
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
        
        elif self.task_type == "multiclass":
            # Multi-class metrics
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='macro', zero_division=0
            )
            
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
                true_labels, pred_labels, average=None, labels=list(range(len(self.vuln_types))), zero_division=0
            )
            
            metrics = {
                "eval_accuracy": float(accuracy),
                "eval_precision": float(precision_weighted),  # Use weighted for main metric
                "eval_recall": float(recall_weighted),
                "eval_f1": float(f1_weighted),
                "eval_precision_macro": float(precision_macro),
                "eval_recall_macro": float(recall_macro),
                "eval_f1_macro": float(f1_macro),
                "eval_num_classes": len(self.vuln_types),
                "eval_total_samples": len(true_labels),
            }
            
            # Add per-class metrics (focus on vulnerable classes)
            vulnerable_count = 0
            vulnerable_recall_sum = 0.0
            
            for vuln_type, idx in self.type_to_id.items():
                if idx < len(precision_per_class):
                    metrics[f"eval_precision_{vuln_type}"] = float(precision_per_class[idx])
                    metrics[f"eval_recall_{vuln_type}"] = float(recall_per_class[idx])
                    metrics[f"eval_f1_{vuln_type}"] = float(f1_per_class[idx])
                    
                    # Track recall for all vulnerable classes combined
                    if vuln_type != 'safe' and support[idx] > 0:
                        vulnerable_recall_sum += recall_per_class[idx] * support[idx]
                        vulnerable_count += support[idx]
            
            # Average recall for vulnerable classes
            if vulnerable_count > 0:
                metrics["eval_recall_vulnerable"] = float(vulnerable_recall_sum / vulnerable_count)
            
            return metrics
    
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
        
        if self.task_type == "binary":
            target_names = ['Safe', 'Vulnerable']
        else:
            target_names = self.vuln_types if hasattr(self, 'vuln_types') else None
        
        return classification_report(
            true_labels, pred_labels,
            target_names=target_names,
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
            
            if 'eval_recall_vulnerable' in metrics:
                print(f"  Recall (Vulnerable): {metrics.get('eval_recall_vulnerable', 0):.4f}")
            
            print(f"  Precision: {metrics.get('eval_precision', 0):.4f}")
            
            # For multiclass, show some per-class metrics
            if 'eval_num_classes' in metrics and metrics['eval_num_classes'] > 2:
                print("  Per-class F1 scores:")
                for key, value in sorted(metrics.items()):
                    if key.startswith('eval_f1_') and not key.endswith('_macro'):
                        class_name = key.replace('eval_f1_', '')
                        if class_name != 'safe':  # Show vulnerable classes
                            print(f"    {class_name}: {value:.4f}")

def generate_metrics_report(trainer, eval_dataset, metrics_calc, output_dir):
    import json
    import os
    
    eval_results = trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    
    confusion_matrix = metrics_calc.compute_confusion_matrix(predictions)
    classification_report = metrics_calc.generate_classification_report(predictions)
    
    # Display confusion matrix based on task type
    if metrics_calc.task_type == "binary":
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Safe  Vulnerable")
        print(f"Actual Safe      {confusion_matrix[0, 0]:<5} {confusion_matrix[0, 1]:<5}")
        print(f"Actual Vulnerable {confusion_matrix[1, 0]:<5} {confusion_matrix[1, 1]:<5}")
    else:
        print("\nConfusion Matrix (showing top classes):")
        # For multiclass, just show the shape and save full matrix
        print(f"Shape: {confusion_matrix.shape}")
        print("Full matrix saved to metrics_report.json")
    
    print("\n" + classification_report)
    
    report = {
        "final_metrics": eval_results,
        "confusion_matrix": confusion_matrix.tolist(),
        "classification_report": classification_report,
        "task_type": metrics_calc.task_type
    }
    
    if hasattr(metrics_calc, 'vuln_types'):
        report["vulnerability_types"] = metrics_calc.vuln_types
    
    with open(os.path.join(output_dir, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def save_confusion_matrix_plot(confusion_matrix, output_dir, metrics_calc=None):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Determine labels based on task type
        if metrics_calc and hasattr(metrics_calc, 'task_type'):
            if metrics_calc.task_type == "binary":
                labels = ['Safe', 'Vulnerable']
            elif hasattr(metrics_calc, 'vuln_types'):
                labels = metrics_calc.vuln_types
                # For many classes, use a larger figure
                if len(labels) > 10:
                    plt.figure(figsize=(12, 10))
                else:
                    plt.figure(figsize=(10, 8))
            else:
                labels = None
        else:
            labels = ['Safe', 'Vulnerable']
            plt.figure(figsize=(8, 6))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()
        
        return cm_path
    except:
        return None