import torch
import torch.nn as nn



class AddPositionEmbedding(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.Tensor(num_patches, dim)) # num_patches 256 , dim 128
    
    def forward(self, x):
        return x + self.pos_embedding  #x shape ([32, 196, 128])    , torch.Size([42, 256, 128])

