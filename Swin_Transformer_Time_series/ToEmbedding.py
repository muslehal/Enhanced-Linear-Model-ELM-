import torch
import torch.nn as nn
from ToPatches import ToPatches
from AddPositionEmbedding import AddPositionEmbedding

class ToEmbedding(nn.Sequential):
    def __init__(self, in_channels, dim, patch_size, num_patches, p_drop=0.):
        super().__init__(
            ToPatches(in_channels, dim, patch_size), # channel 1 , dim 128 , patch 2
            AddPositionEmbedding(dim, num_patches), # num_patches 256
            nn.Dropout(p_drop)
        )
