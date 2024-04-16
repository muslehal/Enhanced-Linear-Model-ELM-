import torch
import torch.nn as nn
from Stage import Stage

class StageStack(nn.Sequential):
    def __init__(self, num_blocks_list, dims, head_dim, shape, window_size, p_drop=0.): # num_blocks_list [4, 4] , dims [128, 128, 256], head_dim 32 , shape(16, 16), window_size 4
        layers = []
        in_dim = dims[0] # in_dim 128
        for num, out_dim in zip(num_blocks_list, dims[1:]):
            layers.append(Stage(num, in_dim, out_dim, head_dim, shape, window_size, p_drop))
            if in_dim != out_dim:
                shape = (shape[0] // 2, shape[1] // 2)
                in_dim = out_dim
        
        super().__init__(*layers)
