import torch
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./config.json")
model = AutoModelForCausalLM.from_config(config)
print(model)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params / 1e9:.3f}B")
print(f"Trainable parameters: {trainable_params / 1e9:.3f}B")