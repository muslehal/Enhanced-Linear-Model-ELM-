import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, shape):
        super().__init__()
        self.shape = shape
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
    
    def forward(self, x):
        x = x.unflatten(1, self.shape).movedim(-1, 1)
        x = F.unfold(x, kernel_size=2, stride=2).movedim(1, -1)
        
        x = self.norm(x)
        x = self.reduction(x)
        return x



