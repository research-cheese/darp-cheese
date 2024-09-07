import torch
from transformers import Conv1D

def get_specific_lora_layer_names(model):
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    return layer_names