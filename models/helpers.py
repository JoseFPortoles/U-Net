import torch
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        init.xavier_uniform_(m.weight.data)