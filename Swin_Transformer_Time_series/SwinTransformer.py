import torch
import torch.nn as nn
from ToEmbedding import ToEmbedding
from StageStack import StageStack
from Head import PredictionHead
from AddPositionEmbedding import AddPositionEmbedding
from ShiftedWindowAttention import ShiftedWindowAttention
from Residual import Residual

class SwinTransformer(nn.Sequential):
    def __init__(self,c_in, classes, image_size,forecast_len,num_blocks_list, dims, head_dim, patch_size, window_size,
                 in_channels=1, emb_p_drop=0., trans_p_drop=0., head_p_drop=0.):
        reduced_size = image_size // patch_size  # reduced_size 32/2 = 16
        #reduced_size = 14 # adjusted for time series data -- improve later
        shape = (reduced_size, reduced_size) #shape (14, 14)
        num_patches = shape[0] * shape[1] # num_patches 196
           
        super().__init__(
            ToEmbedding(in_channels, dims[0], patch_size, num_patches, emb_p_drop), # dims[0]  [128, 128, 256]
            StageStack(num_blocks_list, dims, head_dim, shape, window_size, trans_p_drop),
            #Head(dims[-1], classes, head_p_drop)
            PredictionHead( classes,dims[-1],head_p_drop)
        )
        self.reset_parameters()
        
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
            elif isinstance(m, AddPositionEmbedding):
                nn.init.normal_(m.pos_embedding, mean=0.0, std=0.02)
            elif isinstance(m, ShiftedWindowAttention):
                nn.init.normal_(m.pos_enc, mean=0.0, std=0.02)
            elif isinstance(m, Residual):
                nn.init.zeros_(m.gamma)
    
    def separate_parameters(self):
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear, )
        modules_no_weight_decay = (nn.LayerNorm,)

        for m_name, m in self.named_modules():
            for param_name, param in m.named_parameters():
                full_param_name = f"{m_name}.{param_name}" if m_name else param_name

                if isinstance(m, modules_no_weight_decay):
                    parameters_no_decay.add(full_param_name)
                elif param_name.endswith("bias"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, Residual) and param_name.endswith("gamma"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, AddPositionEmbedding) and param_name.endswith("pos_embedding"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, ShiftedWindowAttention) and param_name.endswith("pos_enc"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, modules_weight_decay):
                    parameters_decay.add(full_param_name)

        ## sanity check
        #assert len(parameters_decay & parameters_no_decay) == 0
        #assert len(parameters_decay) + len(parameters_no_decay) == len(list(model.parameters()))

        return parameters_decay, parameters_no_decay
