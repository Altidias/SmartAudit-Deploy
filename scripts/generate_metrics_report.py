import json
import os
from typing import Dict
import numpy as np

def generate_metrics_report(trainer, eval_dataset, metrics_calc, output_dir):
    
    eval_results = trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    
    confusion_matrix = metrics_calc.compute_confusion_matrix(predictions)
    classification_report = metrics_calc.generate_classification_report(predictions)
    
    pred_labels, true_labels = metrics_calc.extract_predictions(
        predictions.predictions, predictions.label_ids
    )
    
    vulnerability_analysis = analyze_vulnerability_categories(
        pred_labels, true_labels, metrics_calc
    )
    
    report = {
        "framework": "SmartAudit",
        "final_metrics": eval_results,
        "confusion_matrix": confusion_matrix.tolist(),
        "classification_report": classification_report,
        "vulnerability_analysis": vulnerability_analysis,
        "num_classes": metrics_calc.num_classes,
        "model_type": "student"
    }
    
    with open(os.path.join(output_dir, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def analyze_vulnerability_categories(pred_labels, true_labels, metrics_calc):
    analysis = {
        "machine_auditable": {},
        "machine_unauditable": {},
        "overall": {}
    }
    
    # Separate by category
    for vuln_type, vuln_id in metrics_calc.type_to_id.items():
        if vuln_type == 'safe':
            continue
            
        # Get predictions for this vulnerability type
        mask = true_labels == vuln_id
        if mask.sum() == 0:
            continue
            
        correct = (pred_labels[mask] == true_labels[mask]).sum()
        total = mask.sum()
        accuracy = correct / total
        
        category = "machine_auditable" if vuln_type in metrics_calc.machine_auditable else "machine_unauditable"
        analysis[category][vuln_type] = {
            "accuracy": float(accuracy),
            "samples": int(total),
            "correct": int(correct)
        }
    
    # Overall statistics
    for category in ["machine_auditable", "machine_unauditable"]:
        if analysis[category]:
            accuracies = [v["accuracy"] for v in analysis[category].values()]
            analysis["overall"][f"{category}_mean_accuracy"] = float(np.mean(accuracies))
            analysis["overall"][f"{category}_types_count"] = len(analysis[category])
    
    return analysis

def save_confusion_matrix_plot(confusion_matrix, output_dir, metrics_calc=None):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_classes = confusion_matrix.shape[0]
        
        if n_classes > 20:
            # Create a summary view for many classes
            plt.figure(figsize=(15, 12))
            
            # Normalize confusion matrix
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            
            # Show top 20 classes by frequency
            class_freq = confusion_matrix.sum(axis=1)
            top_indices = np.argsort(class_freq)[-20:][::-1]
            
            cm_subset = cm_normalized[top_indices][:, top_indices]
            
            if metrics_calc:
                labels = [metrics_calc.id_to_type.get(i, f"Class {i}") for i in top_indices]
            else:
                labels = [f"Class {i}" for i in top_indices]
            
            sns.heatmap(cm_subset, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Normalized Frequency'})
            
            plt.title('Confusion Matrix (Top 20 Classes)')
        else:
            # Standard view for fewer classes
            plt.figure(figsize=(12, 10))
            labels = metrics_calc.vuln_types if metrics_calc else None
            
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            
            plt.title('Confusion Matrix')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return cm_path
        
    except Exception as e:
        print(f"Failed to save confusion matrix plot: {e}")
        return None