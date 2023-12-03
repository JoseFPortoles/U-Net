import torch
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        init.xavier_uniform_(m.weight.data)

def freeze_encoder(net, freeze: bool=True):
    """Freezes/thaws encoder depending on argument freeze is True/False

    Args:
        net (torch.nn.Module): model
        thaw (bool, optional): encoder freezes (True) or thaws (False). 
            Defaults to True (freeze encoder).
    """    
    for param_name, param in net.named_parameters():
        if param_name.startswith('encoder.'):
            param.requires_grad = not freeze
    return net