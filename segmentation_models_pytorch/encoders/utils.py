def zerocenter(x):
    """x : [B, C, H, W]"""
    return x - x.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)

EPS = 1e-5

import torch.nn.functional as F

def zeronorm(x):
    """x : [B, C, H, W]"""
    return F.layer_norm(x, x.size()[1:], None, None, EPS)