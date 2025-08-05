import torch
from transformers import Trainer
import json
import os

class VulnerabilityDetectionTrainer(Trainer):
    def __init__(self, *args, vulnerable_weight=2.0, tokenizer=None, task_type="binary", **kwargs):
        super().__init__(*args, **kwargs)
        self.vulnerable_weight = vulnerable_weight
        self.tokenizer = tokenizer
        self.task_type = task_type
        
        if task_type == "binary":
            self.vulnerable_token_id = tokenizer.encode("vulnerable", add_special_tokens=False)[0]
            self.safe_token_id = tokenizer.encode("safe", add_special_tokens=False)[0]
        elif task_type == "multiclass":
            # Load vulnerability types
            vuln_types_path = './processed_data_multiclass/vuln_types.json'
            if os.path.exists(vuln_types_path):
                with open(vuln_types_path, 'r') as f:
                    self.vuln_types = json.load(f)
                
                # Create token ID mapping
                self.vuln_token_ids = {}
                for vuln_type in self.vuln_types:
                    tokens = tokenizer.encode(vuln_type, add_special_tokens=False)
                    if tokens:
                        self.vuln_token_ids[vuln_type] = tokens[0]
                
                # Set weights: higher for vulnerable classes, lower for 'safe'
                self.class_weights = {}
                for vuln_type in self.vuln_types:
                    if vuln_type == 'safe':
                        self.class_weights[vuln_type] = 1.0
                    else:
                        self.class_weights[vuln_type] = vulnerable_weight
            else:
                # Fallback to binary if types not found
                print("Warning: vuln_types.json not found, falling back to binary mode")
                self.task_type = "binary"
                self.vulnerable_token_id = tokenizer.encode("vulnerable", add_special_tokens=False)[0]
                self.safe_token_id = tokenizer.encode("safe", add_special_tokens=False)[0]
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        loss = self._weighted_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def _weighted_loss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        vocab_size = self.model.config.vocab_size
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        weights = torch.ones_like(shift_labels, dtype=torch.float)
        
        if self.task_type == "binary":
            # Binary weighting
            weights[shift_labels == self.vulnerable_token_id] = self.vulnerable_weight
        elif self.task_type == "multiclass":
            # Multi-class weighting
            for vuln_type, token_id in self.vuln_token_ids.items():
                weight = self.class_weights.get(vuln_type, 1.0)
                weights[shift_labels == token_id] = weight
        
        weighted_loss = loss.view(shift_labels.size()) * weights
        
        return weighted_loss.mean()