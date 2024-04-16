
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange


    
class ToPatches(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = 14 # 12 or 14 for 512
        #self.proj1 = nn.Linear(28, dim)
        
        self.proj2 = nn.Linear(patch_dim, dim)
        self.norm = nn.LayerNorm(dim)

     
    def forward(self, x):
       # print('topatch',x.shape) #topatch torch.Size([32, 512, 7]), torch.Size([32, 241, 7, 32])
        #x2 = x.permute(0,2, 1)
        x2 = rearrange(x, 'i j m k -> i (j m k)')  
        #print('rearrange',x2.shape)#torch.Size([32, 53984])        x2 = x2[:,0:-20] # 336 -48 , 144 weather + 912, 
        x2 = x2[:,0:]
        #print(x2.shape)#torch.Size([32, 53792]), 
        x2 = rearrange(x2, 'i (j  k) -> i j k', k=14)


        ##print('topatch',x.shape) #topatch torch.Size([32, 512, 7]), torch.Size([32, 241, 7, 32])
        #x3 = rearrange(x, 'i j  k -> i (j k)')  
        ##print('rearrange',x2.shape)#torch.Size([32, 53984])        x2 = x2[:,0:-20] # 336 -48 , 144 weather + 912, 
        #x3 = x3[:,0:-48]
        ##print(x2.shape)#torch.Size([32, 53792]), 
        #x3 = rearrange(x3, 'i (j  k) -> i j k', k=16)

        
        ##print('topatch',x.shape) #topatch torch.Size([32, 512, 7]), torch.Size([32, 241, 7, 32])
        #x4 = rearrange(x, 'i j  k -> i (j k)')  
        ##print('rearrange',x2.shape)#torch.Size([32, 53984])        x2 = x2[:,0:-20] # 336 -48 , 144 weather + 912, 
        #x4 = x4[:,0:-48]
        ##print(x2.shape)#torch.Size([32, 53792]), 
        #x4 = rearrange(x4, 'i (j  k) -> i j k', k=16)
 
        #print('x2',x2.shape)
       # x1 = x.permute(0,1,3,2)
       # ##print('permute',x1.shape)
       # t = rearrange(x1, 'i j m k -> i (j m k)')  
       # #print('rearrange',t.shape)
       # t = t[:,0:-256] # 336 -48 , 144 weather + 912, ill 3584 3564
       # #print(t.shape)
       # x1 = rearrange(t, 'i (j k) -> i j k', k=3) # 9 for 336 , 12 for 439 or 14 for 512, weather 27
       # ##print('x1',x1.shape)
       # #print(x1.shape)
       ## t = torch.stack((x2,x1 ))#,x3,x4
       # #print('stack',t.shape)
       # t = torch.cat((x2,x1  ), dim=2) #,x3,x4
        #print('t',t.shape)
        #x = self.proj1(t)
        x = self.proj2(x2)
        #print('proj',x.shape)
        x = self.norm(x)

        #print('last',x.shape)#last torch.Size([32, 256, 128])
        

        return x
    


    
#class ToPatches(nn.Module):
#    def __init__(self, in_channels, dim, patch_size):
#        super().__init__()
#        self.patch_size = patch_size
#        patch_dim = 1724 # 12 or 14 for 512
#        self.proj = nn.Linear(patch_dim, dim)
#        self.norm = nn.LayerNorm(dim)

        
#    def forward(self, x):
#        #torch.Size([42, 16, 21])
       
#        #x= x.permute(0,2,1)
#        t = rearrange(x, 'i j k -> i (j k)')  

#        t = t[:,0:] # 336 -48 , 192 ELECT-192, 
#        x = rearrange(t, 'i (j k) -> i j k', k=1724) # 9 for 336 , 12 for 439 or 14 for 512 , ELECH 421, weather 27
#        x = self.proj(x)
#        x = self.norm(x)
#        return x
    



       # 2304
        
        
        
        #def forward(self, x):
        #t = rearrange(x, 'i j k -> i (j k)')
        #t = t[:, 0:-1]
    
        ## Pad sequence to make its length divisible by 12
        #seq_len = t.shape[-1]
        #padding_size = -seq_len % 12  # This will give us the required padding size
        #if padding_size > 0:
        #    padding = torch.zeros((t.shape[0], padding_size), dtype=t.dtype, device=t.device)
        #    t = torch.cat([t, padding], dim=-1)
        
        #x = rearrange(t, 'i (j k) -> i j k', k=12)
        #x = self.proj(x)
        #x = self.norm(x)
        #return x
    
#class ToPatches(nn.Module):
#    def __init__(self, in_channels, dim, patch_size):
#        super().__init__()
#        self.patch_size = patch_size # = 2
#        patch_dim = in_channels * patch_size**2 # = 12
#        self.proj = nn.Linear(patch_dim, dim)
#        self.norm = nn.LayerNorm(dim)
    
#    def forward(self, x):
#        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).movedim(1, -1)
#        x = self.proj(x)
#        x = self.norm(x)
#        return x