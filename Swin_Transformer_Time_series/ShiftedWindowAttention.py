import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, head_dim, shape, window_size, shift_size=0):
        super().__init__()
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        
        self.shape = shape
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.unifyheads = nn.Linear(dim, dim)
        
        self.pos_enc = nn.Parameter(torch.Tensor(self.heads, (2 * window_size - 1)**2))
        self.register_buffer("relative_indices", self.get_indices(window_size))
        
        if shift_size > 0:
            self.register_buffer("mask", self.generate_mask(shape, window_size, shift_size))
    
    
    def forward(self, x):
        shift_size, window_size = self.shift_size, self.window_size
        
        x = self.to_windows(x, self.shape, window_size, shift_size) # partition into windows
        
        # self attention
        qkv = self.to_qkv(x).unflatten(-1, (3, self.heads, self.head_dim)).transpose(-2, 1)
        queries, keys, values = qkv.unbind(dim=2)
        
        att = queries @ keys.transpose(-2, -1)
        
        att = att * self.scale + self.get_rel_pos_enc(window_size) # add relative positon encoding
        
        # masking
        if shift_size > 0:
            att = self.mask_attention(att)
        
        att = F.softmax(att, dim=-1)
        
        x = att @ values
        x = x.transpose(1, 2).contiguous().flatten(-2, -1) # move head back
        x = self.unifyheads(x)
        
        x = self.from_windows(x, self.shape, window_size, shift_size) # undo partitioning into windows
        return x
    
    
    def to_windows(self, x, shape, window_size, shift_size):
        x = x.unflatten(1, shape)
        if shift_size > 0:
            x = x.roll((-shift_size, -shift_size), dims=(1, 2))
        x = self.split_windows(x, window_size)
        return x
    
    
    def from_windows(self, x, shape, window_size, shift_size):
        x = self.merge_windows(x, shape, window_size) 
        if shift_size > 0:
            x = x.roll((shift_size, shift_size), dims=(1, 2))
        x = x.flatten(1, 2)
        return x
    
    
    def mask_attention(self, att):
        num_win = self.mask.size(1)
        att = att.unflatten(0, (att.size(0) // num_win, num_win))
        att = att.masked_fill(self.mask, float('-inf'))
        att = att.flatten(0, 1)
        return att
    
    
    def get_rel_pos_enc(self, window_size):
        indices = self.relative_indices.expand(self.heads, -1)
        rel_pos_enc = self.pos_enc.gather(-1, indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (window_size**2, window_size**2))
        return rel_pos_enc
    
    
    # For explanation of mask regions see Figure 4 in the article
    @staticmethod
    def generate_mask(shape, window_size, shift_size):
        region_mask = torch.zeros(1, *shape, 1)
        slices = [slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)]
        
        region_num = 0
        for i in slices:
            for j in slices:
                region_mask[:, i, j, :] = region_num
                region_num += 1

        mask_windows = ShiftedWindowAttention.split_windows(region_mask, window_size).squeeze(-1)
        diff_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = diff_mask != 0
        mask = mask.unsqueeze(1).unsqueeze(0) # add heads and batch dimension
        return mask
    
    
    @staticmethod
    def split_windows(x, window_size):
        #print(f"Original size of x: {x.size()}")

        n_h, n_w = x.size(1) // window_size, x.size(2) // window_size # n_h, n_w = 4
        #print(f"n_h: {n_h}, n_w: {n_w}")
        x = x.unflatten(1, (n_h, window_size)).unflatten(-2, (n_w, window_size)) # split into windows , x = torch.Size([16, 4, 4, 1])
        #print(f"Size of x after unflatten: {x.size()}")

        x = x.transpose(2, 3).flatten(0, 2) # merge batch and window numbers, x = torch.Size([16, 16, 1])
        x = x.flatten(-3, -2)
        return x
    
    
    @staticmethod
    def merge_windows(x, shape, window_size):
        n_h, n_w = shape[0] // window_size, shape[1] // window_size
        b = x.size(0) // (n_h * n_w)
        x = x.unflatten(1, (window_size, window_size))
        x = x.unflatten(0, (b, n_h, n_w)).transpose(2, 3) # separate batch and window numbers
        x = x.flatten(1, 2).flatten(-3, -2) # merge windows
        return x
    
    
    @staticmethod
    def get_indices(window_size):
        x = torch.arange(window_size, dtype=torch.long) # shape =torch.Size([4])
        
        y1, x1, y2, x2 = torch.meshgrid(x, x, x, x, indexing='ij') # y1 , x1 , y2 , x2 = torch.Size([4, 4, 4, 4])
        indices = (y1 - y2 + window_size - 1) * (2 * window_size - 1) + x1 - x2 + window_size - 1  # shape = torch.Size([4, 4, 4, 4])
        indices = indices.flatten() # shape torch.Size([256])
        
        return indices  # shape torch.Size([256])




