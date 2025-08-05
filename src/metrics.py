# metrics.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, Tuple, List
from transformers import TrainerCallback
import json
import os

class SmartAuditMetrics:
    """Metrics calculator for multi-class vulnerability detection"""
    
    def __init__(self, tokenizer, vuln_types_path: str = './processed_data/vuln_types.json'):
        self.tokenizer = tokenizer
        
        # Load vuln types from dataset
        if os.path.exists(vuln_types_path):
            with open(vuln_types_path, 'r') as f:
                self.vuln_types = json.load(f)
        else:
            raise FileNotFoundError(f"Vulnerability types file not found: {vuln_types_path}")
        
        self.num_classes = len(self.vuln_types)
        self.type_to_id = {v: i for i, v in enumerate(self.vuln_types)}
        self.id_to_type = {i: v for v, i in self.type_to_id.items()}
        
        # Get token IDs for each vulnerability type
        self.vuln_token_ids = {}
        for vuln_type in self.vuln_types:
            tokens = tokenizer.encode(vuln_type, add_special_tokens=False)
            if tokens:
                self.vuln_token_ids[vuln_type] = tokens[0]
        
        # Identify machine-auditable vs machine-unauditable vulnerabilities
        # Based on FTSmartAudit paper categorization
        self.machine_auditable = {
            'reentrancy', 'integer-overflow', 'integer-underflow', 
            'access-control', 'unchecked-return', 'uninitialized-storage',
            'locked-ether', 'dos-gas-limit', 'tx-origin', 'timestamp-dependency'
        }
    
    def extract_predictions(self, predictions, labels) -> Tuple[np.ndarray, np.ndarray]:
        """Extract predicted and true labels from model outputs"""
        pred_labels = []
        true_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            found = False
            for j in range(len(label_seq)):
                for vuln_type, token_id in self.vuln_token_ids.items():
                    if label_seq[j] == token_id:
                        # Found a vulnerability type token
                        pred_token = torch.argmax(pred_seq[j]).item()
                        
                        # Find which class the predicted token belongs to
                        pred_class = 0 
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
                "eval_f1_macro": 0.0,
                "eval_recall_vulnerable": 0.0,
            }
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Macro and weighted metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, labels=list(range(self.num_classes)), zero_division=0
        )
        
        metrics = {
            "eval_accuracy": float(accuracy),
            "eval_precision": float(precision_weighted),
            "eval_recall": float(recall_weighted),
            "eval_f1": float(f1_weighted),
            "eval_f1_macro": float(f1_macro),
            "eval_precision_macro": float(precision_macro),
            "eval_recall_macro": float(recall_macro),
            "eval_num_classes": self.num_classes,
            "eval_total_samples": len(true_labels),
        }
        
        # Compute metrics for vulnerable classes (excluding 'safe')
        vulnerable_metrics = self._compute_vulnerable_metrics(
            pred_labels, true_labels, precision_per_class, recall_per_class, 
            f1_per_class, support
        )
        metrics.update(vulnerable_metrics)
        
        # Add per-class metrics for top vulnerability types
        self._add_per_class_metrics(
            metrics, precision_per_class, recall_per_class, 
            f1_per_class, support
        )
        
        return metrics
    
    def _compute_vulnerable_metrics(self, pred_labels, true_labels, 
                                   precision_per_class, recall_per_class, 
                                   f1_per_class, support) -> Dict[str, float]:
        """Compute metrics specifically for vulnerable classes"""
        vulnerable_indices = [i for i, vtype in self.id_to_type.items() if vtype != 'safe']
        
        if not vulnerable_indices:
            return {}
        
        # Filter predictions for vulnerable classes only
        vuln_mask = np.isin(true_labels, vulnerable_indices)
        if vuln_mask.sum() == 0:
            return {"eval_recall_vulnerable": 0.0}
        
        vuln_true = true_labels[vuln_mask]
        vuln_pred = pred_labels[vuln_mask]
        
        # Calculate recall for vulnerable classes
        vuln_recall_sum = 0.0
        vuln_support_sum = 0
        
        for idx in vulnerable_indices:
            if idx < len(support) and support[idx] > 0:
                vuln_recall_sum += recall_per_class[idx] * support[idx]
                vuln_support_sum += support[idx]
        
        metrics = {}
        if vuln_support_sum > 0:
            metrics["eval_recall_vulnerable"] = float(vuln_recall_sum / vuln_support_sum)
        
        # Separate metrics for machine-auditable vs machine-unauditable
        ma_indices = [i for i, vtype in self.id_to_type.items() 
                     if vtype in self.machine_auditable]
        mu_indices = [i for i, vtype in self.id_to_type.items() 
                     if vtype not in self.machine_auditable and vtype != 'safe']
        
        if ma_indices:
            ma_recall = np.mean([recall_per_class[i] for i in ma_indices 
                               if i < len(recall_per_class)])
            metrics["eval_recall_machine_auditable"] = float(ma_recall)
        
        if mu_indices:
            mu_recall = np.mean([recall_per_class[i] for i in mu_indices 
                               if i < len(recall_per_class)])
            metrics["eval_recall_machine_unauditable"] = float(mu_recall)
        
        return metrics
    
    def _add_per_class_metrics(self, metrics, precision_per_class, 
                              recall_per_class, f1_per_class, support):
        """Add per-class metrics for important vulnerability types"""
        # Add metrics for top 10 vulnerability types by support
        vuln_support = [(self.id_to_type[i], support[i], i) 
                       for i in range(len(support)) 
                       if i in self.id_to_type and self.id_to_type[i] != 'safe']
        vuln_support.sort(key=lambda x: x[1], reverse=True)
        
        for vuln_type, sup, idx in vuln_support[:10]:
            if idx < len(precision_per_class):
                metrics[f"eval_f1_{vuln_type}"] = float(f1_per_class[idx])
                metrics[f"eval_recall_{vuln_type}"] = float(recall_per_class[idx])
    
    def generate_classification_report(self, eval_preds) -> str:
        """Generate detailed classification report"""
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        pred_labels, true_labels = self.extract_predictions(predictions, labels)
        
        if len(pred_labels) == 0:
            return "No predictions found"
        
        return classification_report(
            true_labels, pred_labels,
            target_names=self.vuln_types[:min(len(self.vuln_types), 20)],  # Limit display
            digits=4,
            zero_division=0
        )

class SmartAuditCallback(TrainerCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, metrics_calculator):
        super().__init__()
        self.metrics_calculator = metrics_calculator
        self.best_f1_macro = 0.0
        self.best_recall_vulnerable = 0.0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            current_f1 = metrics.get('eval_f1_macro', 0.0)
            current_recall = metrics.get('eval_recall_vulnerable', 0.0)
            
            if current_f1 > self.best_f1_macro:
                self.best_f1_macro = current_f1
                metrics['eval_best_f1_macro'] = self.best_f1_macro
            
            if current_recall > self.best_recall_vulnerable:
                self.best_recall_vulnerable = current_recall
                metrics['eval_best_recall_vulnerable'] = self.best_recall_vulnerable
            
            print(f"\n[Step {state.global_step}] SmartAudit Metrics:")
            print(f"  F1 Macro: {current_f1:.4f} (best: {self.best_f1_macro:.4f})")
            print(f"  Recall (Vulnerable): {current_recall:.4f} (best: {self.best_recall_vulnerable:.4f})")
            print(f"  Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            
            if 'eval_recall_machine_auditable' in metrics:
                print(f"  Machine-Auditable Recall: {metrics['eval_recall_machine_auditable']:.4f}")
            if 'eval_recall_machine_unauditable' in metrics:
                print(f"  Machine-Unauditable Recall: {metrics['eval_recall_machine_unauditable']:.4f}")