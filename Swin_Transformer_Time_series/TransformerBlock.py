import torch
import torch.nn as nn
from ShiftedWindowAttention import ShiftedWindowAttention
from Residual import Residual
from FeedForward import FeedForward

class TransformerBlock(nn.Sequential):
    def __init__(self, dim, head_dim, shape, window_size, shift_size=0, p_drop=0.):
        super().__init__(
            Residual(
                nn.LayerNorm(dim),
                ShiftedWindowAttention(dim, head_dim, shape, window_size, shift_size),
                nn.Dropout(p_drop)
            ),
            Residual(
                nn.LayerNorm(dim),
                FeedForward(dim),
                nn.Dropout(p_drop)
            )
        )


