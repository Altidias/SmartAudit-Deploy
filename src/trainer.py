import torch
from transformers import Trainer

class VulnerabilityDetectionTrainer(Trainer):
    def __init__(self, *args, vulnerable_weight=2.0, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vulnerable_weight = vulnerable_weight
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
        weights[shift_labels == self.vulnerable_token_id] = self.vulnerable_weight
        
        weighted_loss = loss.view(shift_labels.size()) * weights
        
        return weighted_loss.mean()