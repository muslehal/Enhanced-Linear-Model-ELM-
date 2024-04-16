import torch
import torch.nn as nn

class FeedForward(nn.Sequential):
    def __init__(self, dim, mult=4):
        hidden_dim = dim * mult
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)   
        )



