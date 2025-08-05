import torch
from transformers import Trainer
import json
import os

class SmartAuditTrainer(Trainer):
    """Trainer following FTSA methodology"""
    
    def __init__(self, *args, vulnerable_weight=2.5, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vulnerable_weight = vulnerable_weight
        self.tokenizer = tokenizer
        
        vuln_types_path = './processed_data/vuln_types.json'
        with open(vuln_types_path, 'r') as f:
            self.vuln_types = json.load(f)
        
        # Create token ID mapping
        self.vuln_token_ids = {}
        for vuln_type in self.vuln_types:
            tokens = tokenizer.encode(vuln_type, add_special_tokens=False)
            if tokens:
                self.vuln_token_ids[vuln_type] = tokens[0]
        
        # FTSA uses balanced weights with higher weight for vulnerable classes
        self.class_weights = {}
        for vuln_type in self.vuln_types:
            if vuln_type == 'safe':
                self.class_weights[vuln_type] = 1.0
            else:
                self.class_weights[vuln_type] = vulnerable_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss following FTSA approach"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        loss = self._weighted_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def _weighted_loss(self, logits, labels):
        """FTSA weighted cross-entropy loss"""
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        vocab_size = self.model.config.vocab_size
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        weights = torch.ones_like(shift_labels, dtype=torch.float)
        
        # Apply class-specific weights
        for vuln_type, token_id in self.vuln_token_ids.items():
            weight = self.class_weights.get(vuln_type, 1.0)
            weights[shift_labels == token_id] = weight
        
        weighted_loss = loss.view(shift_labels.size()) * weights
        
        # Ignore padding tokens
        mask = shift_labels != -100
        weighted_loss = weighted_loss * mask
        
        return weighted_loss.sum() / mask.sum()