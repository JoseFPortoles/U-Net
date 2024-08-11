import torch
import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def freeze_encoder(net: nn.Module, freeze: bool=True):
    """Freezes/thaws encoder depending on argument freeze is True/False

    Args:
        net (nn.Module): model
        thaw (bool, optional): encoder freezes (True) or thaws (False). 
            Defaults to True (freeze encoder).
    """    
    for param_name, param in net.named_parameters():
        if param_name.startswith('encoder.'):
            param.requires_grad = not freeze
    return net