import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Tuple

def load_model_and_tokenizer(config: Dict) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"\nLoading model: {config['model']['name']}")
    
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora(model: AutoModelForCausalLM, config: Dict) -> AutoModelForCausalLM:
    print("\nApplying LoRA configuration")
    
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
    
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model

def save_model_and_adapters(model, tokenizer, output_dir: str):
    import os
    
    # Save full mdel
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Save LoRA adapters separately
    model.save_pretrained(os.path.join(output_dir, "lora_adapters"))