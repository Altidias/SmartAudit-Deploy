import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Tuple

def load_model_and_tokenizer(config: Dict) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for student training"""
    print(f"\nLoading student model: {config['model']['name']}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=config['model']['trust_remote_code'],
        use_cache=False,
        cache_dir=config['paths']['cache_dir']
    )
    
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        cache_dir=config['paths']['cache_dir']
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora(model: AutoModelForCausalLM, config: Dict) -> AutoModelForCausalLM:
    """Setup LoRA following FTSA configuration"""
    print("\nApplyig LoRA configuration")
    
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
    
    # Log parameter efficiency
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"Parameter efficiency:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Efficiency ratio: {trainable_params/total_params*100:.2f}%")
    
    return model

def save_model_and_adapters(model, tokenizer, output_dir: str):
    """Save student model and LoRA adapters"""
    import os
    
    print("\nSaving student model...")
    
    # Save full model (student model ready for deployment)
    full_model_dir = os.path.join(output_dir, "student_model")
    model.save_pretrained(full_model_dir)
    tokenizer.save_pretrained(full_model_dir)
    
    # Save LoRA adapters separately (for efficient distribution)
    adapters_dir = os.path.join(output_dir, "lora_adapters")
    model.save_pretrained(adapters_dir)
    
    print(f"  Full model saved to: {full_model_dir}")
    print(f"  LoRA adapters saved to: {adapters_dir}")